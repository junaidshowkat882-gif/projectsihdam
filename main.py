"""SIH26161 — Dam-breach flood screening API.

What makes this useful in practice
----------------------------------
* **Terrain comes to the model.** ``POST /api/simulate/auto`` fetches real global
  DEM tiles (and cross-checks them) for a chosen dam, so a full inundation run
  needs no GIS stack, no DEM file and no manual georeferencing.
* **GDAL is optional.** With rasterio installed, ``POST /api/simulate`` also
  accepts an uploaded GeoTIFF. Without it, everything else still runs — the app
  advertises its capabilities and the UI adapts.
* **The output is decision-shaped.** Flooded area, depth statistics, water
  volume, a weir-based peak discharge, severity zones, per-target verdicts,
  mapped-asset exposure and explicit caveats about the run.
* **Enough rope to be honest.** Every response lists the caveats that apply:
  a clipped window, a dam outside the DEM, coarse resolution, nodata gaps.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import exposure
import terrain as terrain_service
from geometry import GridRef, band_geojson, block_mean, is_geographic
from simulation import (
    WEIR_COEFFICIENT,
    DEPTH_BREAKS_M,
    DEPTH_LABELS,
    SimulationResult,
    get_target_metrics,
    simulate_breach,
    world_to_rowcol,
)
from terrain import AffineLike, TerrainError, TerrainRaster

# rasterio/GDAL is a large optional extra: everything except uploaded GeoTIFFs
# works without it, so it is imported defensively rather than required.
try:  # pragma: no cover - depends on the environment
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.io import MemoryFile
    from rasterio.warp import transform_bounds, transform_geom

    RASTERIO_AVAILABLE = True
    RASTERIO_ERROR = ""
except ImportError as exc:  # pragma: no cover - depends on the environment
    rasterio = None  # type: ignore[assignment]
    RASTERIO_AVAILABLE = False
    RASTERIO_ERROR = str(exc)


logger = logging.getLogger("flood.api")

VERSION = "2.0.0"
SCREENING_WARNING = (
    "Screening approximation only; not an engineering-grade hydraulic prediction. "
    "Do not use for evacuation, dam-safety certification or regulatory decisions."
)

BASE_DIR = Path(__file__).resolve().parent
DAM_CSV = BASE_DIR / "dams.csv"
FRONTEND_DIR = BASE_DIR.parent / "frontend"

_GEOTIFF_SUFFIXES = (".tif", ".tiff", ".geotiff")


# Declared outside the dataclass: with slots=True a class-level default is a slot
# descriptor, not the value, so it cannot be read back off the class.
DEFAULT_CORS_ORIGINS: tuple[str, ...] = (
    "http://localhost:8123",
    "http://127.0.0.1:8123",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:8010",
    "http://127.0.0.1:8010",
)


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime configuration, overridable with environment variables."""

    max_grid_size: int = 768
    max_upload_mb: float = 60.0
    cors_origins: tuple[str, ...] = DEFAULT_CORS_ORIGINS
    default_radius_km: float = 30.0
    max_radius_km: float = 40.0
    exposure_enabled: bool = True
    cache_enabled: bool = True
    max_attenuation_m_per_km: float = 5.0

    @classmethod
    def from_env(cls) -> "Settings":
        raw_origins = os.getenv("FLOOD_CORS_ORIGINS", "").strip()
        origins = tuple(part.strip() for part in raw_origins.split(",") if part.strip())

        return cls(
            max_grid_size=int(os.getenv("FLOOD_MAX_GRID_SIZE", "768")),
            max_upload_mb=float(os.getenv("FLOOD_MAX_UPLOAD_MB", "60")),
            default_radius_km=float(os.getenv("FLOOD_DEFAULT_RADIUS_KM", "30")),
            max_radius_km=float(os.getenv("FLOOD_MAX_RADIUS_KM", "40")),
            exposure_enabled=os.getenv("FLOOD_EXPOSURE", "1") not in {"0", "false", "False"},
            cache_enabled=os.getenv("FLOOD_CACHE", "1") not in {"0", "false", "False"},
            max_attenuation_m_per_km=float(os.getenv("FLOOD_MAX_ATTENUATION_M_PER_KM", "15")),
            cors_origins=origins or DEFAULT_CORS_ORIGINS,
        )


settings = Settings.from_env()

app = FastAPI(
    title="SIH26161 Flood Simulation API",
    version=VERSION,
    description=(
        "Screening-level dam-breach inundation. Runs on real global DEM tiles with "
        "no GIS toolchain, or on an uploaded GeoTIFF when GDAL is available."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class Dam(BaseModel):
    dam_id: str
    name: str
    state: str
    latitude: float
    longitude: float
    # Indicative published figures (see README) — edit to match dam records.
    reservoir_level_m: float | None = None
    storage_mcm: float | None = None
    dam_height_m: float | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    capabilities: dict[str, bool]
    max_grid_size: int
    max_upload_mb: float
    default_radius_km: float
    max_radius_km: float


class ApiConfig(BaseModel):
    version: str
    warning: str
    depth_breaks_m: list[float]
    depth_labels: list[str]
    terrain_sources: list[str]
    terrain_provider: str
    terrain_provider_url: str
    upload_available: bool
    upload_unavailable_reason: str
    weir_coefficient: float
    defaults: dict[str, float]
    limits: dict[str, float]


class AutoSimulationRequest(BaseModel):
    """Scenario for the automatic-terrain run: coordinates are WGS84 degrees.

    Reservoir level and release volume default to the dam register, which is why
    the UI can pre-fill them and a caller can still run with just a dam ID.
    """

    dam_id: str = Field(min_length=1, max_length=64)
    release_level_m: float | None = Field(default=None, gt=-500, le=9000)
    release_volume_mcm: float | None = Field(default=None, gt=0, le=100_000)
    breach_head_m: float | None = Field(default=None, gt=0, le=500)
    breach_width_m: float | None = Field(default=None, gt=0, le=5000)
    radius_km: float = Field(default=30.0, ge=2.0, le=60.0)
    # None means "solve the water-surface gradient to match the release volume".
    # The ceiling matches Settings.max_attenuation_m_per_km and the UI slider so a
    # value the interface offers can never be rejected here.
    attenuation_m_per_km: float | None = Field(default=None, ge=0, le=15)
    # Areas of interest are optional: when omitted, two sampling points are
    # placed downstream of the dam along the solved flow direction, so a bare
    # {"dam_id": "DAM001"} still returns a full, populated report.
    area1_lat: float | None = Field(default=None, ge=-90, le=90)
    area1_lon: float | None = Field(default=None, ge=-180, le=180)
    area2_lat: float | None = Field(default=None, ge=-90, le=90)
    area2_lon: float | None = Field(default=None, ge=-180, le=180)
    verify_vertical: bool = True
    include_exposure: bool = False


# ---------------------------------------------------------------------------
# Dam register
# ---------------------------------------------------------------------------
def load_dams() -> pd.DataFrame:
    if not DAM_CSV.exists():
        logger.error("Dam database missing at %s", DAM_CSV)
        raise HTTPException(status_code=500, detail="Dam database not found on the server.")

    frame = pd.read_csv(DAM_CSV)
    required = {"dam_id", "name", "state", "latitude", "longitude"}
    missing = required.difference(frame.columns)
    if missing:
        logger.error("Dam database is missing columns: %s", sorted(missing))
        raise HTTPException(
            status_code=500,
            detail=f"Dam database is missing columns: {sorted(missing)}",
        )
    return frame


def find_dam(dam_id: str) -> pd.Series:
    dams = load_dams()
    matching = dams[dams["dam_id"].astype(str) == str(dam_id)]
    if matching.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Dam ID '{dam_id}' was not found in the register.",
        )
    return matching.iloc[0]


def _optional_float(dam: pd.Series, column: str) -> float | None:
    """Read an optional numeric register column, tolerating blanks."""
    if column not in dam.index:
        return None
    value = dam[column]
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def dam_payload(dam: pd.Series) -> dict:
    return {
        "dam_id": str(dam["dam_id"]),
        "name": str(dam["name"]),
        "state": str(dam["state"]),
        "latitude": float(dam["latitude"]),
        "longitude": float(dam["longitude"]),
        "reservoir_level_m": _optional_float(dam, "reservoir_level_m"),
        "storage_mcm": _optional_float(dam, "storage_mcm"),
        "dam_height_m": _optional_float(dam, "dam_height_m"),
    }


# ---------------------------------------------------------------------------
# Simulation plumbing
# ---------------------------------------------------------------------------
def _downsample_to_budget(
    dem: np.ndarray, transform, max_grid_size: int
) -> tuple[np.ndarray, AffineLike, int]:
    """Bring a mosaic inside the grid budget with a NaN-aware block average."""
    height, width = dem.shape
    factor = max(1, math.ceil(max(height, width) / max_grid_size))
    if factor <= 1:
        affine = AffineLike(transform.a, 0.0, transform.c, 0.0, transform.e, transform.f)
        return dem, affine, 1

    reduced = block_mean(dem, factor)
    affine = AffineLike(
        a=float(transform.a) * factor,
        b=0.0,
        c=float(transform.c),
        d=0.0,
        e=float(transform.e) * factor,
        f=float(transform.f),
    )
    return reduced, affine, factor


def _min_area_m2(result: SimulationResult) -> float:
    """Drop polygons smaller than a handful of cells — they are noise, not zones."""
    return max(4.0 * result.cell_area_m2, 20_000.0)


def _offset_point(
    latitude: float, longitude: float, east_km: float, north_km: float
) -> tuple[float, float]:
    """Move a WGS84 point by a local east/north offset in kilometres."""
    km_per_degree_lat = 111.32
    km_per_degree_lon = max(
        1e-6, 111.32 * math.cos(math.radians(max(-89.5, min(89.5, latitude))))
    )
    return (
        latitude + north_km / km_per_degree_lat,
        longitude + east_km / km_per_degree_lon,
    )


def _derive_targets_from_footprint(
    result: SimulationResult,
    grid: GridRef,
    fractions: tuple[float, float] = (0.35, 0.70),
) -> list[tuple[str, float, float]] | None:
    """Sample the modelled flood along its own reach.

    Two points at roughly a third and two thirds of the way along the inundation
    answer the question a briefing actually asks — *how deep is it, and how fast
    does that decay downstream?* — and, unlike a ray cast from the dam, they do
    not depend on how well the single-cell flow direction happened to resolve.

    Only meaningful for a geographic grid: inverting a projected CRS needs GDAL.
    """
    flooded = result.flooded
    rows, cols = np.nonzero(flooded)
    if rows.size < 8:
        return None

    distances = np.hypot(
        (cols - result.dam_col) * result.cell_size_x_m,
        (rows - result.dam_row) * result.cell_size_y_m,
    )
    furthest = float(distances.max())
    if furthest <= 0:
        return None

    chosen: list[tuple[str, float, float]] = []
    used: set[tuple[int, int]] = set()

    for label, fraction in zip(("Area 1", "Area 2"), fractions):
        order = np.argsort(np.abs(distances - fraction * furthest))
        pick = None
        for index in order[:500]:
            key = (int(rows[index]), int(cols[index]))
            if key not in used:
                pick = int(index)
                break
        if pick is None:
            pick = int(order[0])
        used.add((int(rows[pick]), int(cols[pick])))

        # Cell centre, not corner, so it round-trips back to this exact cell.
        longitude, latitude = grid.lattice_to_native(
            float(cols[pick]) + 0.5, float(rows[pick]) + 0.5
        )
        chosen.append((label, float(latitude), float(longitude)))

    return chosen


def _derive_targets(
    dam_lat: float,
    dam_lon: float,
    direction: tuple[float, float],
    radius_km: float,
    transform,
    dem_crs,
    shape: tuple[int, int],
) -> list[tuple[str, float, float]]:
    """Place two sampling points downstream when the caller supplies none.

    They sit on the flow line at roughly one third and two thirds of the search
    radius, stepped back toward the dam if the window does not reach that far.
    """
    east, north = direction
    magnitude = math.hypot(east, north)
    if magnitude <= 0:
        east, north = 0.0, -1.0
    else:
        east, north = east / magnitude, north / magnitude

    height, width = shape
    placed: list[tuple[str, float, float]] = []

    for label, fraction in (("Area 1", 0.35), ("Area 2", 0.70)):
        chosen: tuple[float, float] | None = None
        for shrink in (1.0, 0.75, 0.5, 0.35, 0.2):
            distance = radius_km * fraction * shrink
            latitude, longitude = _offset_point(
                dam_lat, dam_lon, east * distance, north * distance
            )
            row, col = world_to_rowcol(transform, dem_crs, latitude, longitude)
            if 0 <= row < height and 0 <= col < width:
                chosen = (latitude, longitude)
                break
        if chosen is None:
            chosen = _offset_point(
                dam_lat, dam_lon, east * radius_km * fraction * 0.1, north * radius_km * fraction * 0.1
            )
        placed.append((label, chosen[0], chosen[1]))

    return placed


def _features_bbox(features: list[dict]) -> list[float] | None:
    bbox: list[float] | None = None
    for feature in features:
        polygons = (
            [feature["geometry"]["coordinates"]]
            if feature["geometry"]["type"] == "Polygon"
            else feature["geometry"]["coordinates"]
        )
        for rings in polygons:
            for ring in rings:
                for x, y in ring:
                    if bbox is None:
                        bbox = [x, y, x, y]
                    else:
                        bbox[0] = min(bbox[0], x)
                        bbox[1] = min(bbox[1], y)
                        bbox[2] = max(bbox[2], x)
                        bbox[3] = max(bbox[3], y)
    return bbox


def _reproject_features(features: list[dict], source_crs) -> list[dict]:
    """Move polygon coordinates from a projected DEM CRS into WGS84."""
    if is_geographic(source_crs):
        return features

    for feature in features:
        feature["geometry"] = transform_geom(
            source_crs, "EPSG:4326", feature["geometry"], precision=5
        )
    return features


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
# The headline numbers answer "how much" and the map answers "where"; these
# answer "what does it look like". Everything is read straight off the solved
# grid — no second model run — so a chart can never disagree with the figures.

def _band_areas(result: SimulationResult) -> dict:
    """Area in each severity band, so the headline total can be split up."""
    classes = result.depth_class_grid()
    return {
        "labels": list(DEPTH_LABELS),
        "area_km2": [
            round(
                float(np.count_nonzero(classes == band)) * result.cell_area_m2 / 1e6,
                4,
            )
            for band in range(len(DEPTH_LABELS))
        ],
    }


def _flow_unit_vector(result: SimulationResult) -> tuple[float, float]:
    east, north = result.flow_direction_east_north
    magnitude = math.hypot(east, north)
    if magnitude <= 0:
        return 0.0, -1.0
    return east / magnitude, north / magnitude


def _surface_at(result: SimulationResult, distance_km: float) -> float:
    """Modelled water surface at a given flow distance.

    ``pool_level_m`` is the surface where it leaves the dam and the surface falls
    by ``attenuation_m_per_km`` — the same expression the solver applied to every
    cell.
    """
    return result.pool_level_m - result.attenuation_m_per_km * distance_km


def _longitudinal_profile(
    result: SimulationResult,
    grid: GridRef,
    *,
    geographic: bool,
    samples: int = 161,
) -> dict | None:
    """Ground and modelled water surface from the dam to the edge of the window.

    The single most useful picture of a breach: the pool leaving the dam, the
    solved gradient carrying it downstream, the ground falling away beneath, and
    the exact point where the water runs out.
    """
    height, width = result.depth_m.shape
    east, north = _flow_unit_vector(result)
    cell_w = max(abs(result.cell_size_x_m), 1e-6)
    cell_h = max(abs(result.cell_size_y_m), 1e-6)

    limits: list[float] = []
    if east > 0:
        limits.append((width - 1 - result.dam_col) * cell_w / east)
    elif east < 0:
        limits.append((0 - result.dam_col) * cell_w / east)
    if north > 0:
        limits.append(result.dam_row * cell_h / north)
    elif north < 0:
        limits.append((height - 1 - result.dam_row) * cell_h / (-north))

    reach_m = min([value for value in limits if value > 0], default=0.0)
    if reach_m <= 0:
        reach_m = min(width * cell_w, height * cell_h)

    count = max(2, samples)
    distances: list[float] = []
    latitudes: list[float | None] = []
    longitudes: list[float | None] = []
    terrain_values: list[float | None] = []
    surface_values: list[float | None] = []
    depth_values: list[float | None] = []
    flooded_flags: list[bool] = []

    for index in range(count):
        distance_m = reach_m * index / (count - 1)
        distance_km = distance_m / 1000.0
        col = int(math.floor(result.dam_col + 0.5 + east * distance_m / cell_w))
        row = int(math.floor(result.dam_row + 0.5 - north * distance_m / cell_h))
        inside = 0 <= row < height and 0 <= col < width

        distances.append(round(distance_km, 3))
        surface = _surface_at(result, distance_km)
        surface_values.append(round(surface, 2) if inside else None)

        if inside:
            ground = float(result.elevation_m[row, col])
            flooded = bool(result.flooded[row, col])
            finite = math.isfinite(ground)
            terrain_values.append(round(ground, 1) if finite else None)
            flooded_flags.append(flooded)
            depth_values.append(
                round(float(result.depth_m[row, col]), 2)
                if flooded
                else 0.0
            )
            if geographic:
                longitude, latitude = grid.lattice_to_native(col + 0.5, row + 0.5)
                longitudes.append(round(float(longitude), 5))
                latitudes.append(round(float(latitude), 5))
        else:
            terrain_values.append(None)
            depth_values.append(None)
            flooded_flags.append(False)
            if geographic:
                longitudes.append(None)  # type: ignore[arg-type]
                latitudes.append(None)  # type: ignore[arg-type]

    return {
        "x_label": "Distance downstream from the dam (km)",
        "y_label": "Elevation (m MSL)",
        "reach_km": round(reach_m / 1000.0, 2),
        "gradient_m_per_km": round(result.attenuation_m_per_km, 3),
        "gradient_source": result.attenuation_source,
        "release_level_m": round(result.release_level_m, 2),
        "flow_direction_east_north": list(result.flow_direction_east_north),
        "distance_km": distances,
        "latitude": latitudes if geographic else None,
        "longitude": longitudes if geographic else None,
        "terrain_m": terrain_values,
        "water_surface_m": surface_values,
        "depth_m": depth_values,
        "flooded": flooded_flags,
    }


def _cross_section(
    result: SimulationResult,
    transform,
    dem_crs,
    *,
    latitude: float,
    longitude: float,
    label: str,
    half_width_m: float = 3000.0,
    max_half_width_m: float = 15000.0,
    samples: int = 161,
) -> dict | None:
    """A line cut across the flow at a point of interest.

    Shows the shape of the valley, where the banks are and how far up them the
    water reaches — the question anyone standing at the point actually asks.
    """
    height, width = result.depth_m.shape
    row, col = world_to_rowcol(transform, dem_crs, latitude, longitude)
    if not (0 <= row < height and 0 <= col < width):
        return None

    east, north = _flow_unit_vector(result)
    cell_w = max(abs(result.cell_size_x_m), 1e-6)
    cell_h = max(abs(result.cell_size_y_m), 1e-6)
    across_east, across_north = -north, east  # perpendicular to the flow

    def cell_at(offset_m: float) -> tuple[int, int] | None:
        sample_col = int(math.floor(col + 0.5 + across_east * offset_m / cell_w))
        sample_row = int(
            math.floor(row + 0.5 - across_north * offset_m / cell_h)
        )
        if not (0 <= sample_row < height and 0 <= sample_col < width):
            return None
        return sample_row, sample_col

    def wet_at(offset_m: float) -> bool:
        cell = cell_at(offset_m)
        if cell is None:
            return False
        sample_row, sample_col = cell
        if not math.isfinite(float(result.elevation_m[sample_row, sample_col])):
            return False
        return bool(result.flooded[sample_row, sample_col])

    # Widen until the bank shows on both sides. A fixed window either cuts a
    # flood plain in half or frames nothing but water, and neither answers the
    # question a cross-section exists to answer: how wide is it?
    step_m = max(cell_w, cell_h) * 2.0
    left_edge: float | None = None
    right_edge: float | None = None
    scanned = 0.0
    while scanned < max_half_width_m - step_m:
        scanned += step_m
        if left_edge is None and not wet_at(-scanned):
            left_edge = -scanned
        if right_edge is None and not wet_at(scanned):
            right_edge = scanned
        if left_edge is not None and right_edge is not None:
            break

    lower = max(
        min(
            left_edge if left_edge is not None else -max_half_width_m,
            -half_width_m,
        ),
        -max_half_width_m,
    )
    upper = min(
        max(right_edge if right_edge is not None else max_half_width_m, half_width_m),
        max_half_width_m,
    )

    # Keep both ends inside the raster. A section that starts off the grid would
    # plot with no ground at either edge for no good reason, and "not wet" is true
    # of off-grid cells just as much as of dry land.
    while cell_at(lower) is None and lower < -step_m:
        lower += step_m
    while cell_at(upper) is None and upper > step_m:
        upper -= step_m

    offsets: list[float] = []
    terrain_values: list[float | None] = []
    surface_values: list[float | None] = []
    depth_values: list[float | None] = []
    flooded_flags: list[bool] = []

    count = max(2, samples)
    spacing_km = (upper - lower) / (count - 1) / 1000.0
    wet_samples = 0

    for index in range(count):
        offset_m = lower + (upper - lower) * index / (count - 1)
        offsets.append(round(offset_m / 1000.0, 3))
        cell = cell_at(offset_m)

        if cell is None:
            terrain_values.append(None)
            surface_values.append(None)
            depth_values.append(None)
            flooded_flags.append(False)
            continue

        sample_row, sample_col = cell
        ground = float(result.elevation_m[sample_row, sample_col])
        if not math.isfinite(ground):
            terrain_values.append(None)
            surface_values.append(None)
            depth_values.append(None)
            flooded_flags.append(False)
            continue

        flooded = bool(result.flooded[sample_row, sample_col])
        if flooded:
            wet_samples += 1
        terrain_values.append(round(ground, 1))
        flooded_flags.append(flooded)
        depth_values.append(
            round(float(result.depth_m[sample_row, sample_col]), 2)
            if flooded
            else 0.0
        )

        dx = (sample_col - result.dam_col) * cell_w
        dy = (sample_row - result.dam_row) * cell_h
        surface_values.append(
            round(_surface_at(result, math.hypot(dx, dy) / 1000.0), 2)
        )

    return {
        "label": label,
        "latitude": round(latitude, 5),
        "longitude": round(longitude, 5),
        "x_label": "Distance across the valley (km)",
        "y_label": "Elevation (m MSL)",
        "half_width_km": round(max(abs(lower), abs(upper)) / 1000.0, 2),
        "span_km": round((upper - lower) / 1000.0, 2),
        "flood_width_km": round(wet_samples * spacing_km, 2),
        "offset_km": offsets,
        "terrain_m": terrain_values,
        "water_surface_m": surface_values,
        "depth_m": depth_values,
        "flooded": flooded_flags,
    }


def _build_plots(
    result: SimulationResult,
    transform,
    dem_crs,
    grid: GridRef,
    targets: list,
) -> dict:
    """Assemble the cross-sections and the longitudinal profile for a run."""
    geographic = is_geographic(dem_crs)
    sections = []
    for target in targets:
        section = _cross_section(
            result,
            transform,
            dem_crs,
            latitude=target.latitude,
            longitude=target.longitude,
            label=target.label,
        )
        if section is not None:
            sections.append(section)

    return {
        "profile": _longitudinal_profile(result, grid, geographic=geographic),
        "sections": sections,
    }


def _run_notes(
    result: SimulationResult,
    *,
    window_clipped: bool,
    resolution_m: float | None,
    nodata_fraction: float,
    spikes_rejected: int = 0,
) -> list[dict]:
    """Caveats a reviewer needs before trusting the numbers."""
    notes: list[dict] = []

    if result.release_volume_mcm is None:
        notes.append(
            {
                "level": "info",
                "text": (
                    "No release volume was supplied, so the water surface is held "
                    "flat at the release level — an upper bound, not a volumetric "
                    "result."
                ),
            }
        )
    elif result.attenuation_source == "solved" and result.volume_matched:
        notes.append(
            {
                "level": "info",
                "text": (
                    "Water-surface gradient solved to "
                    f"{result.attenuation_m_per_km:.3f} m/km so the inundation holds "
                    f"the full {result.release_volume_mcm:,.0f} MCM release. The "
                    "gradient is derived from the volume budget, not assumed."
                ),
            }
        )
    elif not result.volume_matched:
        notes.append(
            {
                "level": "warning",
                "text": (
                    f"This analysis window can hold only "
                    f"{result.impounded_volume_mcm:,.0f} MCM of the "
                    f"{result.release_volume_mcm:,.0f} MCM release budget at the "
                    "release level, so the extent is limited by the window. "
                    "Increase the analysis radius."
                ),
            }
        )

    if result.attenuation_source == "user" and result.release_volume_mcm is not None:
        ratio = result.impounded_volume_mcm / max(result.release_volume_mcm, 1e-9)
        notes.append(
            {
                "level": "info",
                "text": (
                    f"Manual gradient of {result.attenuation_m_per_km:.2f} m/km holds "
                    f"{result.impounded_volume_mcm:,.0f} MCM — "
                    f"{ratio * 100:.0f}% of the release budget."
                ),
            }
        )

    if result.snapped_to_nearest_cell:
        notes.append(
            {
                "level": "error" if result.dam_snap_distance_km > 5 else "info",
                "text": (
                    "The dam position was snapped to the nearest valid DEM cell "
                    f"{result.dam_snap_distance_km:.1f} km away."
                ),
            }
        )

    if window_clipped:
        notes.append(
            {
                "level": "warning",
                "text": (
                    "Flooding reached the edge of the analysis window, so the "
                    "extent is clipped. Increase the analysis radius for the full footprint."
                ),
            }
        )

    if resolution_m is not None and resolution_m > 100:
        notes.append(
            {
                "level": "info",
                "text": (
                    f"Terrain resolution is {resolution_m:.0f} m, so narrow valleys "
                    "and small embankments are smoothed out."
                ),
            }
        )

    if spikes_rejected:
        notes.append(
            {
                "level": "info",
                "text": (
                    f"{spikes_rejected} terrain cell{'' if spikes_rejected == 1 else 's'} "
                    "disagreed with their surroundings by more than 600 m — voids or "
                    "corrupt pixels in the source tiles — and were treated as missing "
                    "rather than as ground."
                ),
            }
        )

    if nodata_fraction > 0.05:
        notes.append(
            {
                "level": "warning",
                "text": (
                    f"{nodata_fraction * 100:.0f}% of the analysis window has no "
                    "terrain data and was treated as unmapped."
                ),
            }
        )

    capped_cells = int(
        np.count_nonzero(result.depth_m >= result.max_depth_cap_m - 1e-3)
    )
    if capped_cells:
        notes.append(
            {
                "level": "info",
                "text": (
                    f"Depths are capped at {result.max_depth_cap_m:.0f} m in "
                    f"{capped_cells:,} cells close to the dam, where the "
                    f"{result.release_level_m:.1f} m release level sits far above "
                    "the local ground."
                ),
            }
        )

    if result.flooded_cells < 25:
        notes.append(
            {
                "level": "info",
                "text": (
                    "Very little land floods for this head — check the breach head "
                    "and that the dam coordinates sit on the real structure."
                ),
            }
        )

    return notes


# ---------------------------------------------------------------------------
# 3-D valley grid
# ---------------------------------------------------------------------------
# The frontend's 3-D valley view renders the *modelled* grids, not a cartoon:
# one downsampled copy of the DEM plus the solved depth field, both quantised
# to decimetres so the payload stays small. The 2-D GeoJSON footprint is
# untouched — this block only feeds the Three.js scene.
GRID_3D_MAX = 120  # longest side of the 3-D grid (cells)


def _quantise_grid(values: np.ndarray, scale: float = 10.0) -> list[int]:
    """Round to decimetres and emit row-major ints (NaN -> sentinel -999999)."""
    flat = np.asarray(values, dtype=np.float64).ravel()
    out: list[int] = []
    for value in flat:
        if not math.isfinite(float(value)):
            out.append(-999999)
        else:
            out.append(int(round(float(value) * scale)))
    return out


def _grid3d(result: SimulationResult) -> dict | None:
    """Downsample elevation + depth for the 3-D valley view."""
    elevation = np.asarray(result.elevation_m, dtype=np.float64)
    depth = np.asarray(result.depth_m, dtype=np.float64)
    flooded = np.asarray(result.flooded, dtype=bool)
    if elevation.ndim != 2 or elevation.size == 0:
        return None
    rows, cols = elevation.shape
    step = max(1, math.ceil(max(rows, cols) / GRID_3D_MAX))
    rows_ds = list(range(0, rows, step))
    cols_ds = list(range(0, cols, step))
    elev_ds = elevation[np.ix_(rows_ds, cols_ds)]
    flooded_ds = flooded[np.ix_(rows_ds, cols_ds)]
    depth_ds = np.where(flooded_ds, depth[np.ix_(rows_ds, cols_ds)], 0.0)
    dam_row_ds = min(int(result.dam_row // step), len(rows_ds) - 1)
    dam_col_ds = min(int(result.dam_col // step), len(cols_ds) - 1)
    # Dam height comes from the register when the caller attaches it; the
    # response builder fills it in from the dam payload (None = unknown).
    return {
        "rows": len(rows_ds),
        "cols": len(cols_ds),
        "step_cells": step,
        "cell_size_x_m": round(float(result.cell_size_x_m) * step, 2),
        "cell_size_y_m": round(float(result.cell_size_y_m) * step, 2),
        "dam_row": dam_row_ds,
        "dam_col": dam_col_ds,
        "dam_height_m": None,
        "release_level_m": round(float(result.release_level_m), 2),
        "pool_level_m": round(float(result.pool_level_m), 2),
        "elevation_decim": _quantise_grid(elev_ds),
        "depth_decim": _quantise_grid(depth_ds),
        "scale": 10.0,
        "nodata": -999999,
    }


def _build_response(
    *,
    dam: pd.Series,
    result: SimulationResult,
    targets: list,
    features: list[dict],
    bbox: list[float] | None,
    inputs: dict,
    terrain_block: dict,
    started: float,
    resolution_m: float | None = None,
    nodata_fraction: float = 0.0,
    exposure_block: dict | None = None,
    notes: list[dict] | None = None,
    plots: dict | None = None,
) -> dict:
    grid3d = _grid3d(result)
    if grid3d is not None:
        # Structural height of the dam wall for the 3-D dam block. The model
        # itself never needs it, so a missing register value stays None and the
        # scene falls back to the breach head.
        try:
            height = dam.get("dam_height_m", None)
            grid3d["dam_height_m"] = (
                None if height is None or pd.isna(height) else round(float(height), 1)
            )
        except (TypeError, ValueError, AttributeError):
            grid3d["dam_height_m"] = None
    return {
        "dam": dam_payload(dam),
        "inputs": inputs,
        "summary": {
            "estimated_flooded_area_km2": round(result.estimated_flooded_area_km2, 4),
            "maximum_depth_m": round(result.maximum_depth_m, 2),
            "source_elevation_m": round(result.source_elevation_m, 2),
            "water_surface_m": round(result.pool_level_m, 2),
            "pool_level_m": round(result.pool_level_m, 2),
            "release_level_m": round(result.release_level_m, 2),
            "impounded_volume_mcm": round(result.impounded_volume_mcm, 2),
            "volume_matched": result.volume_matched,
            "flood_volume_m3": round(result.flood_volume_m3, 1),
            "flood_volume_mcm": round(result.flood_volume_m3 / 1e6, 3),
            "peak_discharge_m3s": round(result.peak_discharge_m3s, 1),
            "flooded_cells": result.flooded_cells,
            "depth_stats": result.depth_stats(),
            "depth_band_area": _band_areas(result),
            "grid_rows": int(result.depth_m.shape[0]),
            "grid_columns": int(result.depth_m.shape[1]),
            "cell_size_x_m": round(result.cell_size_x_m, 2),
            "cell_size_y_m": round(result.cell_size_y_m, 2),
            "cell_area_m2": round(result.cell_area_m2, 1),
        },
        "terrain": terrain_block,
        "targets": [target.as_dict() for target in targets],
        "severity_legend": list(DEPTH_LABELS),
        "flood_geojson": {
            "type": "FeatureCollection",
            "features": features,
            "bbox": bbox,
        },
        "exposure": exposure_block,
        "notes": notes or [],
        "plots": plots or {},
        "grid3d": grid3d,
        "processing_ms": round((time.perf_counter() - started) * 1000, 1),
        "warning": SCREENING_WARNING,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=VERSION,
        capabilities={
            "auto_terrain": True,
            "upload_dem": RASTERIO_AVAILABLE,
            "exposure": settings.exposure_enabled,
            "vertical_check": True,
        },
        max_grid_size=settings.max_grid_size,
        max_upload_mb=settings.max_upload_mb,
        default_radius_km=settings.default_radius_km,
        max_radius_km=settings.max_radius_km,
    )


@app.get("/api/config", response_model=ApiConfig)
def config() -> ApiConfig:
    return ApiConfig(
        version=VERSION,
        warning=SCREENING_WARNING,
        depth_breaks_m=list(DEPTH_BREAKS_M),
        depth_labels=list(DEPTH_LABELS),
        terrain_sources=["auto", "upload"] if RASTERIO_AVAILABLE else ["auto"],
        terrain_provider="AWS Terrain Tiles (Mapzen Terrarium)",
        terrain_provider_url=terrain_service.TILE_URL.format(z="{z}", x="{x}", y="{y}"),
        upload_available=RASTERIO_AVAILABLE,
        upload_unavailable_reason=RASTERIO_ERROR,
        weir_coefficient=WEIR_COEFFICIENT,
        defaults={
            "breach_head_m": 30.0,
            "radius_km": settings.default_radius_km,
        },
        limits={
            "max_radius_km": settings.max_radius_km,
            "max_grid_size": float(settings.max_grid_size),
            "max_upload_mb": settings.max_upload_mb,
        },
    )


@app.get("/api/dams", response_model=list[Dam])
def list_dams() -> list[Dam]:
    """Return the dam register for the frontend selector."""
    return [Dam(**record) for record in load_dams().to_dict(orient="records")]


@app.get("/api/dams/{dam_id}", response_model=Dam)
def get_dam(dam_id: str) -> Dam:
    return Dam(**dam_payload(find_dam(dam_id)))


@app.post("/api/simulate/auto")
async def simulate_with_auto_terrain(request: AutoSimulationRequest):
    """Fetch real terrain for the dam and run the screening model on it."""
    started = time.perf_counter()

    dam = await run_in_threadpool(find_dam, request.dam_id)
    dam_lat = float(dam["latitude"])
    dam_lon = float(dam["longitude"])

    register_level = _optional_float(dam, "reservoir_level_m")
    register_storage = _optional_float(dam, "storage_mcm")

    release_level_m = (
        request.release_level_m if request.release_level_m is not None else register_level
    )
    release_volume_mcm = (
        request.release_volume_mcm
        if request.release_volume_mcm is not None
        else register_storage
    )

    if release_level_m is None and request.breach_head_m is None:
        raise HTTPException(
            status_code=422,
            detail=(
                f"No reservoir level is recorded for {dam['name']}, so either a "
                "release level or a breach head is required."
            ),
        )

    level_source = (
        "request"
        if request.release_level_m is not None
        else ("register" if register_level is not None else "derived")
    )
    volume_source = (
        "request"
        if request.release_volume_mcm is not None
        else ("register" if register_storage is not None else "none")
    )

    radius_km = min(request.radius_km, settings.max_radius_km)

    try:
        raster: TerrainRaster = await run_in_threadpool(
            terrain_service.fetch_dem,
            dam_lat,
            dam_lon,
            radius_km,
            use_cache=settings.cache_enabled,
        )
    except TerrainError as exc:
        raise HTTPException(
            status_code=502, detail=f"Terrain acquisition failed: {exc}"
        ) from exc

    dem, transform, downsample_factor = _downsample_to_budget(
        raster.dem, raster.transform, settings.max_grid_size
    )

    result = await run_in_threadpool(
        simulate_breach,
        dem,
        transform,
        raster.crs,
        dam_lat,
        dam_lon,
        release_level_m=release_level_m,
        release_volume_mcm=release_volume_mcm,
        breach_head_m=request.breach_head_m,
        breach_width_m=request.breach_width_m,
        attenuation_m_per_km=request.attenuation_m_per_km,
        max_attenuation_m_per_km=settings.max_attenuation_m_per_km,
    )

    grid = GridRef.from_transform(transform, raster.crs, dam_lat)

    # Each area is resolved independently: a partly supplied pair loses only the
    # half that is missing, rather than being silently thrown away.
    supplied = {
        "Area 1": (request.area1_lat, request.area1_lon),
        "Area 2": (request.area2_lat, request.area2_lon),
    }
    incomplete = [
        label
        for label, (latitude, longitude) in supplied.items()
        if latitude is None or longitude is None
    ]

    derived_by_label: dict[str, tuple[float, float]] = {}
    derived_source = ""
    if incomplete:
        derived = (
            _derive_targets_from_footprint(result, grid)
            if is_geographic(raster.crs)
            else None
        )
        if derived is not None:
            derived_source = "derived from the modelled flood footprint"
        else:
            # Nothing floods, or the CRS is projected: fall back to a ray along
            # the flow direction, which can at least report a dry verdict.
            derived = _derive_targets(
                dam_lat,
                dam_lon,
                result.flow_direction_east_north,
                radius_km,
                transform,
                raster.crs,
                result.depth_m.shape,
            )
            derived_source = "derived along the flow direction"
        derived_by_label = {
            label: (latitude, longitude) for label, latitude, longitude in derived
        }

    target_points: list[tuple[str, float, float]] = []
    for label in ("Area 1", "Area 2"):
        latitude, longitude = supplied[label]
        if latitude is not None and longitude is not None:
            target_points.append((label, float(latitude), float(longitude)))
        else:
            derived_lat, derived_lon = derived_by_label[label]
            target_points.append((label, derived_lat, derived_lon))

    if not incomplete:
        targets_source = "request"
    elif len(incomplete) == len(supplied):
        targets_source = derived_source
    else:
        given = next(label for label in supplied if label not in incomplete)
        targets_source = f"{given} from request; {incomplete[0]} {derived_source}"

    targets = [
        await run_in_threadpool(
            get_target_metrics,
            result,
            transform,
            raster.crs,
            latitude,
            longitude,
            label,
        )
        for label, latitude, longitude in target_points
    ]
    classes = result.depth_class_grid()

    plots = await run_in_threadpool(
        _build_plots, result, transform, raster.crs, grid, targets
    )

    features, bbox = await run_in_threadpool(
        band_geojson,
        classes,
        grid,
        DEPTH_LABELS,
        min_area_m2=_min_area_m2(result),
    )

    exposure_block = None
    if request.include_exposure and settings.exposure_enabled:
        bounds = raster.metadata.get("bounds_wgs84")
        if bounds:
            exposure_block = await run_in_threadpool(
                exposure.analyse_exposure,
                result,
                transform,
                raster.crs,
                tuple(bounds),
                use_cache=settings.cache_enabled,
            )
            if exposure_block is None:
                exposure_block = {
                    "available": False,
                    "reason": "OpenStreetMap Overpass API unavailable or rate-limited.",
                }

    vertical = None
    if request.verify_vertical:
        sample_points = [(dam_lat, dam_lon)] + [
            (target.latitude, target.longitude) for target in targets
        ]
        vertical = await run_in_threadpool(
            terrain_service.vertical_check, raster, sample_points
        )

    resolution_m = float(raster.metadata.get("resolution_m") or 0.0) * downsample_factor
    flooded = result.flooded
    window_clipped = bool(
        flooded[0, :].any()
        or flooded[-1, :].any()
        or flooded[:, 0].any()
        or flooded[:, -1].any()
    )

    terrain_block = {
        "source": "auto",
        "provider": raster.metadata.get("provider"),
        "provider_url": raster.metadata.get("provider_url"),
        "zoom": raster.metadata.get("zoom"),
        "resolution_m": round(resolution_m, 1),
        "native_resolution_m": raster.metadata.get("resolution_m"),
        "downsample_factor": downsample_factor,
        "mosaic_rows": int(raster.dem.shape[0]),
        "mosaic_columns": int(raster.dem.shape[1]),
        "tiles_requested": raster.metadata.get("tiles_requested"),
        "tiles_failed": raster.metadata.get("tiles_failed"),
        "bounds_wgs84": raster.metadata.get("bounds_wgs84"),
        "elevation_min_m": raster.metadata.get("elevation_min_m"),
        "elevation_max_m": raster.metadata.get("elevation_max_m"),
        "nodata_fraction": raster.metadata.get("nodata_fraction"),
        "spikes_rejected": raster.metadata.get("spikes_rejected"),
        "radius_km": radius_km,
        "vertical_check": vertical,
    }

    return _build_response(
        dam=dam,
        result=result,
        targets=targets,
        features=features,
        bbox=bbox,
        inputs={
            "terrain_source": "auto",
            "release_level_m": round(result.release_level_m, 2),
            "release_level_source": level_source,
            "release_volume_mcm": (
                None if result.release_volume_mcm is None else round(result.release_volume_mcm, 2)
            ),
            "release_volume_source": volume_source,
            "pool_level_m": round(result.pool_level_m, 2),
            "volume_matched": result.volume_matched,
            "impounded_volume_mcm": round(result.impounded_volume_mcm, 2),
            "breach_head_m": round(result.breach_head_m, 2),
            "breach_width_m": round(result.breach_width_m, 2),
            "attenuation_m_per_km": round(result.attenuation_m_per_km, 3),
            "attenuation_source": result.attenuation_source,
            "flow_direction_east_north": list(result.flow_direction_east_north),
            "radius_km": radius_km,
            "targets_source": targets_source,
            "dam_snapped_to_nearest_cell": result.snapped_to_nearest_cell,
            "dam_snap_distance_km": result.dam_snap_distance_km,
        },
        terrain_block=terrain_block,
        started=started,
        resolution_m=resolution_m,
        nodata_fraction=float(raster.metadata.get("nodata_fraction") or 0.0),
        exposure_block=exposure_block,
        plots=plots,
        notes=_run_notes(
            result,
            window_clipped=window_clipped,
            resolution_m=resolution_m,
            nodata_fraction=float(raster.metadata.get("nodata_fraction") or 0.0),
            spikes_rejected=int(raster.metadata.get("spikes_rejected") or 0),
        ),
    )


@app.post("/api/simulate")
async def simulate_with_uploaded_dem(
    dam_id: Annotated[str, Form(min_length=1, max_length=64)],
    dem_file: Annotated[UploadFile, File()],
    area1_lat: Annotated[float, Form(ge=-90, le=90)],
    area1_lon: Annotated[float, Form(ge=-180, le=180)],
    area2_lat: Annotated[float, Form(ge=-90, le=90)],
    area2_lon: Annotated[float, Form(ge=-180, le=180)],
    release_level_m: Annotated[float | None, Form(gt=-500, le=9000)] = None,
    release_volume_mcm: Annotated[float | None, Form(gt=0, le=100_000)] = None,
    breach_head_m: Annotated[float | None, Form(gt=0, le=500)] = None,
    breach_width_m: Annotated[float | None, Form(gt=0, le=5000)] = None,
    attenuation_m_per_km: Annotated[float | None, Form(ge=0, le=15)] = None,
):
    """Run the screening model on a user-supplied GeoTIFF DEM."""
    if not RASTERIO_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=(
                "Uploaded GeoTIFF support needs rasterio (GDAL), which is not "
                f"installed on this server: {RASTERIO_ERROR}. "
                "Use the automatic global-DEM run instead, or install the full stack."
            ),
        )

    started = time.perf_counter()

    dam = await run_in_threadpool(find_dam, dam_id)
    dam_lat = float(dam["latitude"])
    dam_lon = float(dam["longitude"])

    register_level = _optional_float(dam, "reservoir_level_m")
    register_storage = _optional_float(dam, "storage_mcm")
    resolved_level = release_level_m if release_level_m is not None else register_level
    resolved_volume = (
        release_volume_mcm if release_volume_mcm is not None else register_storage
    )

    if resolved_level is None and breach_head_m is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "Provide either a release level or a breach head for the uploaded DEM run."
            ),
        )

    filename = (dem_file.filename or "").lower()
    if not filename.endswith(_GEOTIFF_SUFFIXES):
        raise HTTPException(status_code=400, detail="Upload a GeoTIFF DEM (.tif or .tiff).")

    cap_bytes = int(settings.max_upload_mb * 1024 * 1024)
    dem_bytes = await dem_file.read()
    if not dem_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(dem_bytes) > cap_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"DEM is larger than the {settings.max_upload_mb:g} MB limit.",
        )

    try:
        dem, source_crs, transform, metadata = await run_in_threadpool(
            read_dem, dem_bytes
        )
        result = await run_in_threadpool(
            simulate_breach,
            dem,
            transform,
            source_crs,
            dam_lat,
            dam_lon,
            release_level_m=resolved_level,
            release_volume_mcm=resolved_volume,
            breach_head_m=breach_head_m,
            breach_width_m=breach_width_m,
            attenuation_m_per_km=attenuation_m_per_km,
            max_attenuation_m_per_km=settings.max_attenuation_m_per_km,
        )
        targets = [
            await run_in_threadpool(
                get_target_metrics,
                result,
                transform,
                source_crs,
                latitude,
                longitude,
                label,
            )
            for label, latitude, longitude in (
                ("Area 1", area1_lat, area1_lon),
                ("Area 2", area2_lat, area2_lon),
            )
        ]
        grid = GridRef.from_transform(transform, source_crs, dam_lat)
        features, bbox = await run_in_threadpool(
            band_geojson,
            result.depth_class_grid(),
            grid,
            DEPTH_LABELS,
            min_area_m2=_min_area_m2(result),
        )
        features = await run_in_threadpool(_reproject_features, features, source_crs)
        bbox = _features_bbox(features)
        plots = await run_in_threadpool(
            _build_plots, result, transform, source_crs, grid, targets
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - surfaced to the client
        logger.exception("Simulation failed for dam %s", dam_id)
        raise HTTPException(status_code=400, detail=f"Simulation failed: {exc}") from exc

    flooded = result.flooded
    window_clipped = bool(
        flooded[0, :].any() or flooded[-1, :].any() or flooded[:, 0].any() or flooded[:, -1].any()
    )

    return _build_response(
        dam=dam,
        result=result,
        targets=targets,
        features=features,
        bbox=bbox,
        inputs={
            "terrain_source": "upload",
            "filename": dem_file.filename,
            "release_level_m": round(result.release_level_m, 2),
            "release_level_source": "request" if release_level_m is not None else "register",
            "release_volume_mcm": (
                None if result.release_volume_mcm is None else round(result.release_volume_mcm, 2)
            ),
            "release_volume_source": (
                "request" if release_volume_mcm is not None else "register"
            ),
            "pool_level_m": round(result.pool_level_m, 2),
            "volume_matched": result.volume_matched,
            "impounded_volume_mcm": round(result.impounded_volume_mcm, 2),
            "breach_head_m": round(result.breach_head_m, 2),
            "breach_width_m": round(result.breach_width_m, 2),
            "attenuation_m_per_km": round(result.attenuation_m_per_km, 3),
            "attenuation_source": result.attenuation_source,
            "dam_snapped_to_nearest_cell": result.snapped_to_nearest_cell,
            "dam_snap_distance_km": result.dam_snap_distance_km,
        },
        terrain_block={
            "source": "upload",
            "provider": "user-supplied GeoTIFF",
            "resolution_m": round(result.cell_size_x_m, 1),
            "downsample_factor": metadata.get("downsample_factor"),
            "mosaic_rows": metadata.get("source_height"),
            "mosaic_columns": metadata.get("source_width"),
            "bounds_wgs84": metadata.get("bounds_wgs84"),
            "crs": metadata.get("crs"),
            "nodata_fraction": None,
            "vertical_check": None,
        },
        started=started,
        resolution_m=result.cell_size_x_m,
        plots=plots,
        notes=_run_notes(
            result,
            window_clipped=window_clipped,
            resolution_m=result.cell_size_x_m,
            nodata_fraction=0.0,
        ),
    )


@app.get("/api/version")
def version() -> dict[str, str]:
    return {"version": VERSION}


@app.post("/api/cache/clear")
def clear_caches() -> dict[str, int]:
    """Drop cached terrain tiles and Overpass responses (handy before a demo)."""
    return {
        "terrain_tiles_removed": terrain_service.clear_cache(),
        "exposure_responses_removed": exposure.clear_cache(),
    }


# ---------------------------------------------------------------------------
# Raster reading (uploaded GeoTIFFs, GDAL only)
# ---------------------------------------------------------------------------
def read_dem(dem_bytes: bytes) -> tuple[np.ndarray, object, object, dict]:
    """Decode an uploaded GeoTIFF into a downsampled float32 DEM."""
    from rasterio.errors import RasterioIOError

    try:
        with MemoryFile(dem_bytes) as memory_file:
            with memory_file.open() as src:
                if src.crs is None:
                    raise HTTPException(
                        status_code=400,
                        detail="The uploaded DEM has no CRS / projection information.",
                    )
                if src.count < 1:
                    raise HTTPException(status_code=400, detail="The uploaded raster has no bands.")

                factor = max(
                    1, math.ceil(max(src.width, src.height) / settings.max_grid_size)
                )
                out_height = max(1, math.ceil(src.height / factor))
                out_width = max(1, math.ceil(src.width / factor))

                dem = src.read(
                    1,
                    out_shape=(out_height, out_width),
                    resampling=Resampling.bilinear,
                ).astype("float32")

                valid_mask = (
                    src.read_masks(
                        1,
                        out_shape=(out_height, out_width),
                        resampling=Resampling.nearest,
                    )
                    > 0
                )

                # The mask already drops nodata, but an explicit value test still
                # catches rasters whose nodata lies inside the mask.
                if src.nodata is not None:
                    valid_mask &= ~np.isclose(dem, src.nodata)

                dem[~valid_mask] = np.nan

                transform = src.transform * rasterio.Affine.scale(
                    src.width / out_width, src.height / out_height
                )

                west, south, east, north = transform_bounds(
                    src.crs, "EPSG:4326", *src.bounds, densify_pts=21
                )

                metadata = {
                    "crs": str(src.crs),
                    "source_width": int(src.width),
                    "source_height": int(src.height),
                    "downsample_factor": int(factor),
                    "has_overviews": bool(src.overviews(1)),
                    "nodata": None if src.nodata is None else float(src.nodata),
                    "bounds_wgs84": [west, south, east, north],
                }

                if not np.isfinite(dem).any():
                    raise HTTPException(
                        status_code=400,
                        detail="The DEM contains no valid elevation values.",
                    )

                return dem, src.crs, transform, metadata
    except HTTPException:
        raise
    except RasterioIOError as exc:
        raise HTTPException(
            status_code=400, detail=f"Could not read the uploaded DEM: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Static frontend (served last so it never shadows an API route)
# ---------------------------------------------------------------------------
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:  # pragma: no cover - only when the frontend folder is absent
    @app.get("/")
    def missing_frontend() -> dict[str, str]:
        return {
            "status": "api-only",
            "detail": f"Frontend directory not found at {FRONTEND_DIR}.",
            "docs": "/docs",
        }