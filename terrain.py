"""Global DEM acquisition from free, key-less public terrain services.

Two independent public services are used:

``AWS Terrain Tiles`` (the Mapzen "Terrarium" RGB PNG set)
    The mosaic that actually gets modelled. Elevation is encoded losslessly in
    the R/G/B channels as ``(R * 256 + G + B / 256) - 32768`` metres, so decoding
    is exact and needs no GIS stack — just zlib and NumPy.

``Open-Meteo Elevation API`` (Copernicus DEM GLO-90)
    An independent vertical cross-check: sampling the same coordinates from a
    *different* dataset tells the user how much to trust the mosaic instead of
    asking them to take it on faith.

Both are free and need no API key. Tiles are cached on disk, so repeat runs of
the same scenario are fast and work offline.

Georeferencing note
-------------------
Terrarium tiles are Web Mercator. Longitude is linear in pixels but latitude is
not, so the mosaic is linearised between its exactly-known north and south
edges. Inside a 20-40 km analysis box the residual error is a small fraction of
one pixel, and :func:`sample_open_meteo` is available to prove it.
"""

from __future__ import annotations

import json
import math
import struct
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
ELEVATION_API = "https://api.open-meteo.com/v1/elevation"

TILE_SIZE = 256
NODATA_M = -32768.0
# A single cell that disagrees with its 3×3 neighbourhood by more than this is a
# void or a corrupt decode, not ground: at 30–130 m/px real terrain never steps
# that far in one cell, whereas the Terrarium files carry occasional lone pixels
# hundreds of metres below the terrain around them. 600 m clears the spikes seen
# in practice (700–1200 m of deviation) while leaving genuine sea cliffs alone.
SPIKE_THRESHOLD_M = 600.0
MAX_TILES = 48
# Longest side of the mosaic. At 640 px a 20 km radius resolves zoom 11 (~65 m/px)
# instead of zoom 10 (~130 m/px), which is the difference between seeing a valley
# and seeing a smear.
TARGET_PIXELS = 640
MIN_ZOOM, MAX_ZOOM = 7, 14
FETCH_WORKERS = 8
REQUEST_TIMEOUT_S = 25

USER_AGENT = "SIH26161-flood-screening/1.2 (screening prototype; +https://sih.gov.in)"
CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "terrarium"
CACHE_VERSION = 1

# WGS84 spherical constants used for on-the-fly metric conversions.
M_PER_DEG_LAT = 110_574.0
M_PER_DEG_LON = 111_320.0
EQUATOR_M_PER_PIXEL_Z0 = 156_543.03392


class TerrainError(RuntimeError):
    """Raised when terrain data cannot be obtained for a location."""


@dataclass(slots=True)
class AffineLike:
    """Minimal ``rasterio.Affine`` work-alike: pixel (col, row) -> (x, y)."""

    a: float
    b: float
    c: float
    d: float
    e: float
    f: float

    def __iter__(self):
        return iter((self.a, self.b, self.c, self.d, self.e, self.f))


@dataclass(slots=True)
class TerrainRaster:
    """A decoded DEM mosaic plus everything needed to georeference it."""

    dem: np.ndarray
    transform: AffineLike
    crs: str
    latitude: float
    longitude: float
    radius_km: float
    metadata: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.dem.shape

    @property
    def resolution_m(self) -> float:
        return float(self.metadata.get("resolution_m", 0.0))


# ---------------------------------------------------------------------------
# Web Mercator tile maths
# ---------------------------------------------------------------------------
def meters_per_pixel(zoom: int, latitude: float) -> float:
    return EQUATOR_M_PER_PIXEL_Z0 * math.cos(math.radians(latitude)) / (2**zoom)


def lonlat_to_world_px(latitude: float, longitude: float, zoom: int) -> tuple[float, float]:
    """WGS84 -> global pixel coordinates at ``zoom`` (256 px tiles)."""
    world_size = TILE_SIZE * (2**zoom)
    x = (longitude + 180.0) / 360.0 * world_size

    lat = min(max(latitude, -85.05112878), 85.05112878)
    sin_lat = math.sin(math.radians(lat))
    y = (0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * world_size
    return x, y


def world_px_to_lonlat(px: float, py: float, zoom: int) -> tuple[float, float]:
    world_size = TILE_SIZE * (2**zoom)
    longitude = px / world_size * 360.0 - 180.0
    n = math.pi * (1.0 - 2.0 * py / world_size)
    latitude = math.degrees(math.atan(math.sinh(n)))
    return latitude, longitude


def analysis_bounds(
    latitude: float, longitude: float, radius_km: float
) -> tuple[float, float, float, float]:
    """Rough WGS84 box around a point for a given ground radius (west, south, east, north)."""
    d_lat = radius_km / (M_PER_DEG_LAT / 1000.0)
    d_lon = radius_km / (
        (M_PER_DEG_LON / 1000.0) * max(math.cos(math.radians(latitude)), 1e-6)
    )
    return (
        longitude - d_lon,
        latitude - d_lat,
        longitude + d_lon,
        latitude + d_lat,
    )


def tile_span(latitude: float, radius_km: float, zoom: int) -> tuple[int, int]:
    """Number of tiles (across, down) needed to cover the analysis box."""
    west, south, east, north = analysis_bounds(latitude, latitude, radius_km)
    x0, y0 = lonlat_to_world_px(north, west, zoom)
    x1, y1 = lonlat_to_world_px(south, east, zoom)
    return (
        int(math.floor(x1 / TILE_SIZE)) - int(math.floor(x0 / TILE_SIZE)) + 1,
        int(math.floor(y1 / TILE_SIZE)) - int(math.floor(y0 / TILE_SIZE)) + 1,
    )


def choose_zoom(
    latitude: float,
    radius_km: float,
    target_pixels: int = TARGET_PIXELS,
    max_tiles: int = MAX_TILES,
) -> tuple[int, float]:
    """Pick the finest zoom whose mosaic stays within the pixel and tile budget."""
    latitude = min(max(latitude, -60.0), 60.0)

    for zoom in range(MAX_ZOOM, MIN_ZOOM - 1, -1):
        mpp = meters_per_pixel(zoom, latitude)
        pixels_across = (2.0 * radius_km * 1000.0) / mpp
        if pixels_across > target_pixels:
            continue
        across, down = tile_span(latitude, radius_km, zoom)
        if across * down <= max_tiles:
            return zoom, mpp

    zoom = MIN_ZOOM
    return zoom, meters_per_pixel(zoom, latitude)


# ---------------------------------------------------------------------------
# PNG decoding (8-bit, non-interlaced — what Terrarium ships)
# ---------------------------------------------------------------------------
def decode_png(data: bytes) -> np.ndarray:
    """Decode an 8-bit non-interlaced PNG into an (H, W, C) uint8 array."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise TerrainError("Terrain tile is not a PNG.")

    position = 8
    idat = bytearray()
    width = height = None
    bit_depth = color_type = interlace = None

    while position + 8 <= len(data):
        (length,) = struct.unpack(">I", data[position : position + 4])
        chunk_type = data[position + 4 : position + 8]
        chunk = data[position + 8 : position + 8 + length]
        position += 12 + length

        if chunk_type == b"IHDR":
            (
                width,
                height,
                bit_depth,
                color_type,
                _compression,
                _filter,
                interlace,
            ) = struct.unpack(">IIBBBBB", chunk)
        elif chunk_type == b"IDAT":
            idat += chunk
        elif chunk_type == b"IEND":
            break

    if width is None or height is None:
        raise TerrainError("PNG is missing its IHDR chunk.")
    if bit_depth != 8 or interlace != 0:
        raise TerrainError(
            f"Unsupported PNG (bit depth {bit_depth}, interlace {interlace})."
        )

    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color_type)
    if channels is None:
        raise TerrainError(f"Unsupported PNG colour type {color_type}.")

    raw = zlib.decompress(bytes(idat))
    stride = width * channels
    bpp = channels

    out = np.empty((height, stride), dtype=np.uint8)
    previous = np.zeros(stride, dtype=np.uint8)
    offset = 0

    for row in range(height):
        filter_type = raw[offset]
        offset += 1
        line = np.frombuffer(raw, dtype=np.uint8, count=stride, offset=offset)
        offset += stride

        if filter_type == 0:  # None
            current = line.astype(np.uint8)
        elif filter_type == 2:  # Up
            current = ((line.astype(np.int32) + previous.astype(np.int32)) & 0xFF).astype(
                np.uint8
            )
        else:
            values = line.astype(np.int32)
            if filter_type == 1:  # Sub
                for i in range(bpp, stride):
                    values[i] = (values[i] + values[i - bpp]) & 0xFF
            elif filter_type == 3:  # Average
                for i in range(stride):
                    left = values[i - bpp] if i >= bpp else 0
                    values[i] = (values[i] + ((left + int(previous[i])) >> 1)) & 0xFF
            elif filter_type == 4:  # Paeth
                for i in range(stride):
                    left = int(values[i - bpp]) if i >= bpp else 0
                    up = int(previous[i])
                    upper_left = int(previous[i - bpp]) if i >= bpp else 0
                    estimate = left + up - upper_left
                    pa = abs(estimate - left)
                    pb = abs(estimate - up)
                    pc = abs(estimate - upper_left)
                    if pa <= pb and pa <= pc:
                        predictor = left
                    elif pb <= pc:
                        predictor = up
                    else:
                        predictor = upper_left
                    values[i] = (values[i] + predictor) & 0xFF
            else:
                raise TerrainError(f"Unsupported PNG filter {filter_type}.")
            current = values.astype(np.uint8)

        out[row] = current
        previous = current

    return out.reshape(height, width, channels)


def decode_terrarium(png: np.ndarray) -> np.ndarray:
    """Terrarium RGB -> elevation in metres (float32, 1 m quantisation)."""
    if png.shape[2] < 3:
        raise TerrainError("Terrarium tiles must be RGB.")

    rgb = png[:, :, :3].astype(np.float32)
    elevation = rgb[:, :, 0] * 256.0 + rgb[:, :, 1] + rgb[:, :, 2] / 256.0 - 32768.0
    return elevation.astype(np.float32)


# ---------------------------------------------------------------------------
# Tile acquisition
# ---------------------------------------------------------------------------
def _fetch_bytes(url: str, timeout: int = REQUEST_TIMEOUT_S) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _cache_path(zoom: int, tile_x: int, tile_y: int) -> Path:
    return CACHE_DIR / f"v{CACHE_VERSION}" / str(zoom) / str(tile_x) / f"{tile_y}.npy"


def fetch_tile(zoom: int, tile_x: int, tile_y: int, use_cache: bool = True) -> np.ndarray:
    """Return one tile's elevation array, using the on-disk cache when possible."""
    cache_path = _cache_path(zoom, tile_x, tile_y)
    if use_cache and cache_path.exists():
        try:
            cached = np.load(cache_path)
            if cached.shape == (TILE_SIZE, TILE_SIZE):
                return cached
        except (OSError, ValueError):
            cache_path.unlink(missing_ok=True)

    url = TILE_URL.format(z=zoom, x=tile_x, y=tile_y)
    payload = _fetch_bytes(url)
    elevation = decode_terrarium(decode_png(payload))
    elevation[elevation <= NODATA_M + 1.0] = np.nan

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp.npy")
        np.save(temporary, elevation)
        temporary.replace(cache_path)

    return elevation


def reject_spikes(
    dem: np.ndarray, threshold_m: float = SPIKE_THRESHOLD_M
) -> tuple[np.ndarray, int]:
    """Null lone cells that are wildly out of step with their neighbourhood.

    Global DEMs ship voids, and occasionally a decode that survives the explicit
    nodata test — a lone pixel hundreds of metres below the ground around it. Left
    in place it distorts the reported elevation range, and if the flood fill ever
    reaches it, the volume budget too. The filter is applied to the cropped
    analysis window rather than per tile, so it also works on tiles already in the
    on-disk cache.

    Returns the cleaned copy and how many cells were rejected.
    """
    height, width = dem.shape
    if height < 3 or width < 3:
        return dem, 0

    padded = np.pad(dem, 1, mode="edge")
    neighbourhood = np.stack(
        [
            padded[dy : dy + height, dx : dx + width]
            for dy in range(3)
            for dx in range(3)
        ],
        axis=-1,
    )
    with np.errstate(invalid="ignore", over="ignore"):
        centre = np.nanmedian(neighbourhood, axis=-1)
        deviation = np.abs(dem - centre)

    # NaN cells carry no deviation and are already nodata.
    spikes = np.nan_to_num(deviation, nan=0.0) > threshold_m
    if not spikes.any():
        return dem, 0

    cleaned = dem.copy()
    cleaned[spikes] = np.nan
    return cleaned, int(spikes.sum())


def fetch_dem(
    latitude: float,
    longitude: float,
    radius_km: float,
    *,
    use_cache: bool = True,
    max_tiles: int = MAX_TILES,
) -> TerrainRaster:
    """Download, mosaic and georeference a DEM around a point."""
    if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
        raise TerrainError(f"Coordinates out of range: {latitude}, {longitude}.")
    if not (0.5 <= radius_km <= 120.0):
        raise TerrainError("Analysis radius must be between 0.5 km and 120 km.")
    if abs(latitude) >= 85.0:
        raise TerrainError("Polar latitudes are not covered by the terrain tiles.")

    zoom, mpp = choose_zoom(latitude, radius_km, max_tiles=max_tiles)

    west, south, east, north = analysis_bounds(latitude, longitude, radius_km)
    x0, y0 = lonlat_to_world_px(north, west, zoom)
    x1, y1 = lonlat_to_world_px(south, east, zoom)

    tile_x0, tile_x1 = int(math.floor(x0 / TILE_SIZE)), int(math.floor(x1 / TILE_SIZE))
    tile_y0, tile_y1 = int(math.floor(y0 / TILE_SIZE)), int(math.floor(y1 / TILE_SIZE))

    tile_count = (tile_x1 - tile_x0 + 1) * (tile_y1 - tile_y0 + 1)
    if tile_count > max_tiles:
        raise TerrainError(
            f"Analysis window needs {tile_count} tiles (limit {max_tiles}); "
            "reduce the radius."
        )

    mosaic_height = (tile_y1 - tile_y0 + 1) * TILE_SIZE
    mosaic_width = (tile_x1 - tile_x0 + 1) * TILE_SIZE
    mosaic = np.full((mosaic_height, mosaic_width), np.nan, dtype=np.float32)

    jobs = [
        (tile_x, tile_y)
        for tile_y in range(tile_y0, tile_y1 + 1)
        for tile_x in range(tile_x0, tile_x1 + 1)
    ]

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        futures = {
            pool.submit(fetch_tile, zoom, tile_x, tile_y, use_cache): (tile_x, tile_y)
            for tile_x, tile_y in jobs
        }
        for future, (tile_x, tile_y) in futures.items():
            try:
                tile = future.result()
            except (urllib.error.URLError, OSError, TerrainError, ValueError) as exc:
                failures.append(f"{zoom}/{tile_x}/{tile_y}: {exc}")
                continue

            row = (tile_y - tile_y0) * TILE_SIZE
            col = (tile_x - tile_x0) * TILE_SIZE
            mosaic[row : row + TILE_SIZE, col : col + TILE_SIZE] = tile

    if failures and len(failures) > max(1, len(jobs) // 3):
        raise TerrainError(
            "Too many terrain tiles failed to download "
            f"({len(failures)}/{len(jobs)}): {failures[0]}"
        )

    # Crop to the exact analysis box.
    crop_x0 = int(math.floor(x0)) - tile_x0 * TILE_SIZE
    crop_y0 = int(math.floor(y0)) - tile_y0 * TILE_SIZE
    crop_x1 = int(math.ceil(x1)) - tile_x0 * TILE_SIZE
    crop_y1 = int(math.ceil(y1)) - tile_y0 * TILE_SIZE

    crop_x0 = max(0, min(crop_x0, mosaic_width - 1))
    crop_y0 = max(0, min(crop_y0, mosaic_height - 1))
    crop_x1 = max(crop_x0 + 2, min(crop_x1, mosaic_width))
    crop_y1 = max(crop_y0 + 2, min(crop_y1, mosaic_height))

    dem = np.ascontiguousarray(mosaic[crop_y0:crop_y1, crop_x0:crop_x1], dtype=np.float32)
    dem, spikes_rejected = reject_spikes(dem)
    height, width = dem.shape

    # Exact geographic edges of the crop, then a linearised transform inside it.
    west_lon = world_px_to_lonlat(tile_x0 * TILE_SIZE + crop_x0, 0, zoom)[1]
    east_lon = world_px_to_lonlat(tile_x0 * TILE_SIZE + crop_x1, 0, zoom)[1]
    north_lat = world_px_to_lonlat(0, tile_y0 * TILE_SIZE + crop_y0, zoom)[0]
    south_lat = world_px_to_lonlat(0, tile_y0 * TILE_SIZE + crop_y1, zoom)[0]

    delta_lon = (east_lon - west_lon) / width
    delta_lat = (north_lat - south_lat) / height

    transform = AffineLike(
        a=delta_lon,
        b=0.0,
        c=west_lon,
        d=0.0,
        e=-delta_lat,
        f=north_lat,
    )

    valid = dem[np.isfinite(dem)]
    metadata = {
        "provider": "AWS Terrain Tiles (Mapzen Terrarium, SRTM/NED composite)",
        "provider_url": TILE_URL.format(z="{z}", x="{x}", y="{y}"),
        "zoom": zoom,
        "tiles_requested": len(jobs),
        "tiles_failed": len(failures),
        "tile_failures": failures[:5],
        "crs": "EPSG:4326",
        "width": width,
        "height": height,
        "resolution_m": round(
            meters_per_pixel(zoom, (north_lat + south_lat) / 2.0), 1
        ),
        "resolution_m_lon": round(delta_lon * M_PER_DEG_LON * math.cos(math.radians(latitude)), 1),
        "resolution_m_lat": round(delta_lat * M_PER_DEG_LAT, 1),
        "bounds_wgs84": [west_lon, south_lat, east_lon, north_lat],
        "elevation_min_m": None if valid.size == 0 else round(float(valid.min()), 1),
        "elevation_max_m": None if valid.size == 0 else round(float(valid.max()), 1),
        "nodata_fraction": round(float(1.0 - valid.size / dem.size), 4),
        "spikes_rejected": spikes_rejected,
        "cached": use_cache,
        "georeferencing": "linearised Web Mercator within the analysis box",
    }

    if valid.size == 0:
        raise TerrainError("The terrain tiles returned no valid elevation data.")

    return TerrainRaster(
        dem=dem,
        transform=transform,
        crs="EPSG:4326",
        latitude=latitude,
        longitude=longitude,
        radius_km=radius_km,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Independent vertical cross-check
# ---------------------------------------------------------------------------
def sample_open_meteo(points: list[tuple[float, float]], timeout: int = 15) -> list[float]:
    """Return Copernicus-DEM elevations (metres) for WGS84 ``(lat, lon)`` points."""
    if not points:
        return []
    if len(points) > 100:
        raise TerrainError("Open-Meteo accepts at most 100 points per request.")

    latitudes = ",".join(f"{lat:.5f}" for lat, _ in points)
    longitudes = ",".join(f"{lon:.5f}" for _, lon in points)
    url = f"{ELEVATION_API}?latitude={latitudes}&longitude={longitudes}"

    payload = json.loads(_fetch_bytes(url, timeout=timeout).decode("utf-8"))

    if isinstance(payload, dict):
        return [float(value) for value in payload.get("elevation", [])]

    return [float(entry["elevation"]) for entry in payload]


def sample_grid(raster: TerrainRaster, points: list[tuple[float, float]]) -> list[float | None]:
    """Nearest-cell elevation lookup from a mosaic (NaN where unavailable)."""
    height, width = raster.dem.shape
    transform = raster.transform
    values: list[float | None] = []

    for latitude, longitude in points:
        col = int(math.floor((longitude - transform.c) / transform.a))
        row = int(math.floor((latitude - transform.f) / transform.e))
        if 0 <= row < height and 0 <= col < width:
            value = float(raster.dem[row, col])
            values.append(value if math.isfinite(value) else None)
        else:
            values.append(None)

    return values


def vertical_check(
    raster: TerrainRaster, points: list[tuple[float, float]]
) -> dict | None:
    """Compare the mosaic against Open-Meteo at the given points."""
    try:
        reference = sample_open_meteo(points)
    except (urllib.error.URLError, OSError, ValueError, KeyError, TerrainError):
        return None

    mosaic = sample_grid(raster, points)
    deltas = [
        abs(a - b)
        for a, b in zip(mosaic, reference)
        if a is not None and b is not None
    ]
    if not deltas:
        return None

    return {
        "reference": "Open-Meteo Elevation API (Copernicus DEM GLO-90)",
        "samples": len(deltas),
        "points": [
            {
                "latitude": round(lat, 5),
                "longitude": round(lon, 5),
                "mosaic_m": None if a is None else round(a, 1),
                "reference_m": round(b, 1),
                "delta_m": None if a is None else round(abs(a - b), 1),
            }
            for (lat, lon), a, b in zip(points, mosaic, reference)
        ],
        "mean_abs_delta_m": round(sum(deltas) / len(deltas), 1),
        "max_abs_delta_m": round(max(deltas), 1),
    }


def clear_cache() -> int:
    """Delete cached tiles and return how many files were removed."""
    if not CACHE_DIR.exists():
        return 0
    removed = 0
    for path in CACHE_DIR.rglob("*.npy"):
        path.unlink(missing_ok=True)
        removed += 1
    return removed


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description="Fetch a terrain mosaic and report its provenance."
    )
    parser.add_argument("--lat", type=float, default=31.411)
    parser.add_argument("--lon", type=float, default=76.432)
    parser.add_argument("--radius", type=float, default=20.0)
    parser.add_argument("--no-cache", action="store_true")
    arguments = parser.parse_args()

    started = time.perf_counter()
    raster = fetch_dem(
        arguments.lat, arguments.lon, arguments.radius, use_cache=not arguments.no_cache
    )
    elapsed = time.perf_counter() - started

    print(f"Mosaic  : {raster.shape[1]} x {raster.shape[0]} px")
    print(f"Source  : {raster.metadata['provider']}")
    print(f"Zoom    : {raster.metadata['zoom']}  ({raster.metadata['resolution_m']} m/px)")
    print(f"Bounds  : {raster.metadata['bounds_wgs84']}")
    print(f"Elev    : {raster.metadata['elevation_min_m']} .. {raster.metadata['elevation_max_m']} m")
    print(f"Fetch   : {elapsed:.1f} s ({raster.metadata['tiles_requested']} tiles)")

    check = vertical_check(raster, [(arguments.lat, arguments.lon)])
    if check:
        print(
            f"Check   : mosaic {check['points'][0]['mosaic_m']} m vs "
            f"reference {check['points'][0]['reference_m']} m "
            f"(delta {check['points'][0]['delta_m']} m)"
        )