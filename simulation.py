"""Screening-level dam-break inundation model.

This is deliberately *not* a hydraulic solver: there is no unsteady routing, no
hydrograph, no breach evolution and no sediment transport. It answers one
screening question — *given a reservoir released from its full level, how much
connected land lies below the resulting pool, and how deep is the water there?* —
and it answers it in a way that cannot violate the reservoir's own volume budget.

Model
-----
1. **Start level.** The pool starts at ``release_level_m`` (normally the dam's
   full reservoir level from the register). If no level is supplied, the historic
   simplest assumption is used instead: local terrain at the dam + breach head.
2. **Connectivity.** A breadth-first flood fill from the dam marks every cell
   that water can *reach* from the dam without crossing higher ground. Isolated
   low basins far away are never reported.
3. **Volume budget.** The pool is then lowered to the level that impounds exactly
   ``release_volume_mcm`` of water inside that connected region. This is what
   keeps a 9.6 km³ reservoir from "flooding" 40 km³ of valley — a flat pool at the
   release level is an upper bound, and the solver reports how much of the budget
   the terrain could actually hold.
4. **Attenuation.** Optionally the water surface decays with flow distance, a
   crude stand-in for energy loss so the far field is not held at dam level.
5. **Discharge.** Peak breach discharge uses the broad-crested (critical flow)
   weir relation ``Q = (8/27) × b × √g × H^1.5``.

Everything is reported in SI units (m, m², m³, m³/s, km²). No hard dependency on
rasterio: coordinates use plain arithmetic for geographic DEMs and rasterio is
imported lazily only when a projected CRS must be reprojected.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from geometry import GridRef, is_geographic

GRAVITY = 9.80665

# Broad-crested weir coefficient for critical flow through a rectangular notch.
WEIR_COEFFICIENT = 8.0 / 27.0

# Depth bands used by the API to draw severity zones on the map.
DEPTH_BREAKS_M: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0)
DEPTH_LABELS: tuple[str, ...] = ("<0.5 m", "0.5-1 m", "1-2 m", "2-5 m", ">5 m")


@dataclass(slots=True)
class SimulationResult:
    """Output of :func:`simulate_breach`."""

    flooded: np.ndarray
    depth_m: np.ndarray
    elevation_m: np.ndarray
    pool_level_m: float
    release_level_m: float
    level_source: str
    source_elevation_m: float
    release_volume_mcm: float | None
    impounded_volume_mcm: float
    volume_matched: bool
    attenuation_source: str
    dam_row: int
    dam_col: int
    dam_latitude: float
    dam_longitude: float
    cell_area_m2: float
    cell_size_x_m: float
    cell_size_y_m: float
    breach_head_m: float
    breach_width_m: float
    attenuation_m_per_km: float
    max_depth_cap_m: float
    flow_direction_east_north: tuple[float, float] = (0.0, 0.0)
    snapped_to_nearest_cell: bool = False
    dam_snap_distance_km: float = 0.0
    flooded_cells: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.flooded_cells = int(np.count_nonzero(self.flooded))

    @property
    def estimated_flooded_area_km2(self) -> float:
        return self.flooded_cells * self.cell_area_m2 / 1_000_000.0

    @property
    def flood_volume_m3(self) -> float:
        """Water held in the inundation (Σ depth × cell area)."""
        if self.flooded_cells == 0:
            return 0.0
        return float(np.nansum(self.depth_m[self.flooded])) * self.cell_area_m2

    @property
    def peak_discharge_m3s(self) -> float:
        """Broad-crested weir estimate of the breach peak discharge."""
        return (
            WEIR_COEFFICIENT
            * self.breach_width_m
            * math.sqrt(GRAVITY)
            * self.breach_head_m**1.5
        )

    @property
    def maximum_depth_m(self) -> float:
        if self.flooded_cells == 0:
            return 0.0
        return float(np.nanmax(self.depth_m[self.flooded]))

    @property
    def mean_depth_m(self) -> float:
        if self.flooded_cells == 0:
            return 0.0
        return float(np.nanmean(self.depth_m[self.flooded]))

    def depth_percentile_m(self, percentile: float) -> float:
        if self.flooded_cells == 0:
            return 0.0
        return float(np.nanpercentile(self.depth_m[self.flooded], percentile))

    def depth_class_grid(self) -> np.ndarray:
        """Integer grid of severity classes (-1 outside the flood)."""
        classes = np.full(self.depth_m.shape, -1, dtype=np.int16)
        depths = self.depth_m[self.flooded]
        if depths.size == 0:
            return classes

        classes[self.flooded] = np.digitize(depths, DEPTH_BREAKS_M).astype(np.int16)
        return classes

    def depth_stats(self) -> dict[str, float]:
        return {
            "mean_m": round(self.mean_depth_m, 3),
            "median_m": round(self.depth_percentile_m(50), 3),
            "p95_m": round(self.depth_percentile_m(95), 3),
            "max_m": round(self.maximum_depth_m, 3),
        }


@dataclass(slots=True)
class TargetMetrics:
    """Screening metrics at a single user-supplied point of interest."""

    latitude: float
    longitude: float
    label: str = ""
    row: int | None = None
    col: int | None = None
    inside_grid: bool = False
    ground_elevation_m: float | None = None
    flood_depth_m: float = 0.0
    inundated: bool = False
    distance_from_dam_km: float | None = None

    @property
    def severity(self) -> str:
        if not self.inundated:
            return "dry"
        depth = self.flood_depth_m
        if depth < DEPTH_BREAKS_M[0]:
            return "minor"
        if depth < DEPTH_BREAKS_M[1]:
            return "moderate"
        if depth < DEPTH_BREAKS_M[2]:
            return "severe"
        return "extreme"

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "latitude": round(self.latitude, 6),
            "longitude": round(self.longitude, 6),
            "row": self.row,
            "col": self.col,
            "inside_grid": self.inside_grid,
            "ground_elevation_m": (
                None
                if self.ground_elevation_m is None
                else round(self.ground_elevation_m, 2)
            ),
            "flood_depth_m": round(self.flood_depth_m, 2),
            "inundated": self.inundated,
            "severity": self.severity,
            "distance_from_dam_km": (
                None
                if self.distance_from_dam_km is None
                else round(self.distance_from_dam_km, 2)
            ),
        }


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _to_dem_coordinates(dem_crs, latitude: float, longitude: float) -> tuple[float, float]:
    """WGS84 degrees -> DEM native coordinates (reprojecting only if required)."""
    if is_geographic(dem_crs):
        return float(longitude), float(latitude)

    try:
        from rasterio.warp import transform as warp_transform  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ValueError(
            "Reprojecting coordinates into a projected DEM requires rasterio; "
            "use a geographic DEM (EPSG:4326) instead."
        ) from exc

    xs, ys = warp_transform("EPSG:4326", dem_crs, [longitude], [latitude])
    return float(xs[0]), float(ys[0])


def world_to_rowcol(transform, dem_crs, latitude: float, longitude: float) -> tuple[int, int]:
    """Convert a WGS84 lat/lon into raster (row, col)."""
    x, y = _to_dem_coordinates(dem_crs, latitude, longitude)
    col = int(math.floor((x - transform.c) / transform.a))
    row = int(math.floor((y - transform.f) / transform.e))
    return row, col


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    return 2.0 * radius_km * math.asin(min(1.0, math.sqrt(a)))


def _nearest_valid_cell(
    valid: np.ndarray,
    row: int,
    col: int,
    cell_w_m: float,
    cell_h_m: float,
) -> tuple[int, int, bool, float]:
    """Closest valid cell to (row, col), with a 'moved' flag and the shift in km.

    Distances use metres (not cells) so non-square pixels cannot bias the search.
    """
    height, width = valid.shape
    if 0 <= row < height and 0 <= col < width and valid[row, col]:
        return row, col, False, 0.0

    rows, cols = np.nonzero(valid)
    if rows.size == 0:
        raise ValueError("The DEM contains no valid (non-nodata) cells.")

    east_m = (cols - col) * cell_w_m
    north_m = (rows - row) * cell_h_m
    best = int(np.argmin(east_m**2 + north_m**2))
    snap_km = math.hypot(east_m[best], north_m[best]) / 1000.0
    return int(rows[best]), int(cols[best]), True, snap_km


# ---------------------------------------------------------------------------
# Flood fill and level solve
# ---------------------------------------------------------------------------
def downstream_direction(
    dem: np.ndarray,
    valid: np.ndarray,
    row: int,
    col: int,
    cell_w_m: float,
    cell_h_m: float,
    radius: int = 2,
) -> tuple[float, float]:
    """Unit vector of steepest descent at the dam, as (east, north) components.

    Searched over a small neighbourhood rather than a single cell so that the
    direction does not hinge on one noisy pixel of the dam structure.
    """
    height, width = dem.shape
    centre = float(dem[row, col])
    best_east = best_north = 0.0
    best_gradient = 0.0

    for d_row in range(-radius, radius + 1):
        for d_col in range(-radius, radius + 1):
            if d_row == 0 and d_col == 0:
                continue

            sample_row = row + d_row
            sample_col = col + d_col
            if not (0 <= sample_row < height and 0 <= sample_col < width):
                continue
            if not valid[sample_row, sample_col]:
                continue

            drop = centre - float(dem[sample_row, sample_col])
            if drop <= 0:
                continue

            east = d_col * cell_w_m
            north = -d_row * cell_h_m  # row index increases southwards

            # Steepest gradient, not deepest neighbour: on a uniform slope every
            # diagonal has the same drop as the straight line, and an axis-aligned
            # direction is the correct one.
            gradient = drop / math.hypot(east, north)
            if gradient <= best_gradient:
                continue

            best_gradient = gradient
            best_east = east
            best_north = north

    if best_gradient <= 0:
        return 0.0, -1.0  # flat or rising terrain: assume flow to the south

    magnitude = math.hypot(best_east, best_north)
    if magnitude <= 0:
        return 0.0, -1.0
    return best_east / magnitude, best_north / magnitude


def downstream_mask(
    shape: tuple[int, int],
    row: int,
    col: int,
    direction: tuple[float, float],
    cell_w_m: float,
    cell_h_m: float,
    margin_cells: float = 1.5,
) -> np.ndarray:
    """Cells in the half-plane the dam releases into.

    A dam-break flood occupies the reach *below* the dam, not the reservoir basin
    behind it — which matters on real terrain, because satellite DEMs render an
    existing reservoir as a flat sheet just under the full reservoir level and it
    would otherwise be counted as freshly flooded.
    """
    height, width = shape
    rows, columns = np.mgrid[0:height, 0:width]

    east_m = (columns - col) * cell_w_m
    north_m = (row - rows) * cell_h_m

    projection = east_m * direction[0] + north_m * direction[1]
    return projection >= -margin_cells * max(cell_w_m, cell_h_m)


def _flood_fill(
    dem: np.ndarray,
    valid: np.ndarray,
    start: tuple[int, int],
    surface_m: float,
    cell_w_m: float,
    cell_h_m: float,
    allowed: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Connectivity-aware fill up to ``surface_m``; returns (mask, flow distance km).

    Breadth-first distances are non-decreasing, so the greedy expansion always
    finds the shortest path and therefore the highest local water surface.
    """
    height, width = dem.shape
    mask = np.zeros((height, width), dtype=bool)
    distance_km = np.full((height, width), np.inf, dtype="float32")

    row, col = start
    mask[row, col] = True
    distance_km[row, col] = 0.0

    queue: deque[tuple[int, int]] = deque([(row, col)])
    neighbours = ((1, 0, cell_h_m), (-1, 0, cell_h_m), (0, 1, cell_w_m), (0, -1, cell_w_m))

    while queue:
        current_row, current_col = queue.popleft()
        current_distance = float(distance_km[current_row, current_col])

        for d_row, d_col, step_m in neighbours:
            next_row = current_row + d_row
            next_col = current_col + d_col
            if not (0 <= next_row < height and 0 <= next_col < width):
                continue
            if mask[next_row, next_col] or not valid[next_row, next_col]:
                continue
            if allowed is not None and not allowed[next_row, next_col]:
                continue
            if dem[next_row, next_col] >= surface_m:
                continue

            mask[next_row, next_col] = True
            distance_km[next_row, next_col] = current_distance + step_m / 1000.0
            queue.append((next_row, next_col))

    return mask, distance_km


def _attenuated_volume(
    gap_m: np.ndarray, distance_km: np.ndarray, attenuation: float, cell_area_m2: float
) -> float:
    """Volume held by a surface that starts ``gap_m`` above the ground and falls
    linearly with flow distance."""
    depth = np.maximum(gap_m - attenuation * distance_km, 0.0)
    return float(np.sum(depth)) * cell_area_m2


def _solve_attenuation(
    gap_m: np.ndarray,
    distance_km: np.ndarray,
    target_volume_m3: float,
    cell_area_m2: float,
    max_attenuation: float,
) -> tuple[float, bool]:
    """Solve the water-surface gradient that makes the pool hold ``target_volume_m3``.

    A static pool held at the reservoir level is only an upper bound: a released
    volume spreading downstream *must* fall away from the dam. So rather than
    inventing a gradient, this solves for it — the surface starts at the release
    level with ``gap_m`` of water over each cell and decays at ``α`` metres per
    kilometre until the inundation holds exactly the reservoir's release volume.

    Volume is continuous and strictly decreasing in ``α``, so a bisection is both
    exact enough and fast (no flood fill inside the loop).

    Returns ``(attenuation_m_per_km, volume_matched)``. ``volume_matched`` is
    False when even the flat pool cannot hold the requested volume, which means
    the analysis window is too small for the reservoir — a fact the caller
    reports instead of hiding.
    """
    if gap_m.size == 0:
        return 0.0, False

    flat_volume = _attenuated_volume(gap_m, distance_km, 0.0, cell_area_m2)
    if target_volume_m3 >= flat_volume:
        return 0.0, False

    # Bracket the root. The gradient needed to dry out the *most stubborn* cell
    # (largest depth-to-distance ratio) is the point where the volume bottoms out
    # at whatever the dam cell itself still holds — beyond that, more gradient
    # changes nothing. Using the smallest ratio instead would fail to bracket.
    moving = distance_km > 0
    if not moving.any():
        return 0.0, False

    ratios = gap_m[moving] / distance_km[moving]
    upper = float(np.max(ratios))
    if not math.isfinite(upper) or upper <= 0:
        return 0.0, False

    lower = 0.0
    for _ in range(48):
        middle = (lower + upper) / 2.0
        if _attenuated_volume(gap_m, distance_km, middle, cell_area_m2) > target_volume_m3:
            lower = middle
        else:
            upper = middle

    attenuation = (lower + upper) / 2.0
    if attenuation > max_attenuation:
        return max_attenuation, False

    # Self-check rather than trust: the bisection can converge on the bracket edge
    # when the budget is smaller than the dam cell's own water.
    achieved = _attenuated_volume(gap_m, distance_km, attenuation, cell_area_m2)
    matched = abs(achieved - target_volume_m3) <= 0.02 * target_volume_m3
    return attenuation, matched


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------
def simulate_breach(
    dem: np.ndarray,
    transform,
    dem_crs,
    dam_latitude: float,
    dam_longitude: float,
    *,
    release_level_m: float | None = None,
    release_volume_mcm: float | None = None,
    breach_head_m: float | None = None,
    breach_width_m: float | None = None,
    attenuation_m_per_km: float | None = None,
    max_depth_cap_m: float = 150.0,
    max_attenuation_m_per_km: float = 15.0,
    downstream_only: bool = True,
) -> SimulationResult:
    """Run the screening inundation model.

    Args:
        dem: 2-D float array of terrain elevations (NaN = nodata).
        transform: Affine transform of ``dem`` (``rasterio.Affine`` compatible).
        dem_crs: CRS of ``dem`` (a string such as ``"EPSG:4326"`` is fine).
        dam_latitude: Dam latitude in WGS84 degrees.
        dam_longitude: Dam longitude in WGS84 degrees.
        release_level_m: Reservoir surface at the moment of breaching (m MSL).
        release_volume_mcm: Volume available to drain, in million m³. When given,
            the pool is lowered until it impounds exactly this much water.
        breach_head_m: Water depth over the breach, used for the discharge
            estimate. Defaults to the depth of the release level above the dam
            terrain.
        breach_width_m: Breach width for the discharge estimate; defaults to the
            breach head (the usual first-cut assumption).
        attenuation_m_per_km: Decay of the water surface with flow distance. When
            omitted and a release volume is supplied, the gradient is *solved* so
            the inundation holds exactly that volume.
        max_depth_cap_m: Clip for physically implausible depths.
        max_attenuation_m_per_km: Ceiling on the solved gradient.
        downstream_only: Scope the flood to the reach below the dam, excluding the
            reservoir basin behind it.

    Returns:
        A :class:`SimulationResult` with the flooded mask, depth field and metrics.
    """
    dem = np.asarray(dem, dtype="float32")
    if dem.ndim != 2 or dem.size == 0:
        raise ValueError("DEM must be a non-empty 2-D array.")
    if release_level_m is None and breach_head_m is None:
        raise ValueError("Provide either a release level or a breach head.")
    if release_volume_mcm is not None and release_volume_mcm <= 0:
        raise ValueError("The release volume must be greater than zero.")

    height, width = dem.shape
    valid = np.isfinite(dem)

    grid = GridRef.from_transform(transform, dem_crs, dam_latitude)
    cell_w_m = grid.cell_width_m
    cell_h_m = grid.cell_height_m
    cell_area_m2 = cell_w_m * cell_h_m

    row, col = world_to_rowcol(transform, dem_crs, dam_latitude, dam_longitude)
    row, col, snapped, snap_km = _nearest_valid_cell(valid, row, col, cell_w_m, cell_h_m)

    source_elevation = float(dem[row, col])

    if release_level_m is not None:
        release_level = float(release_level_m)
        level_source = "release level"
    else:
        release_level = source_elevation + float(breach_head_m)
        level_source = "dam terrain + breach head"

    if release_level <= source_elevation:
        raise ValueError(
            "The release level is at or below the terrain at the dam, so nothing "
            "can be released. Check the level and the dam coordinates."
        )

    resolved_head = (
        float(breach_head_m)
        if breach_head_m is not None
        else max(0.1, release_level - source_elevation)
    )
    resolved_width = resolved_head if breach_width_m is None else float(breach_width_m)

    # 1. Everything the water can physically reach from the dam, at the release level,
    #    on the downstream side of the structure.
    if downstream_only:
        flow = downstream_direction(dem, valid, row, col, cell_w_m, cell_h_m)
        allowed = downstream_mask(
            (height, width), row, col, flow, cell_w_m, cell_h_m
        )
    else:
        flow = (0.0, 0.0)
        allowed = None

    candidate, distance_km = _flood_fill(
        dem, valid, (row, col), release_level, cell_w_m, cell_h_m, allowed
    )

    # 2. Resolve the water-surface gradient. A static pool at the release level is
    #    only an upper bound; given a release volume, the gradient is solved so the
    #    inundation holds exactly that volume.
    volume_matched = True
    if attenuation_m_per_km is not None:
        resolved_attenuation = float(attenuation_m_per_km)
        attenuation_source = "user"
    elif release_volume_mcm is None:
        resolved_attenuation = 0.0
        attenuation_source = "none"
    else:
        resolved_attenuation, volume_matched = _solve_attenuation(
            release_level - dem[candidate],
            distance_km[candidate],
            float(release_volume_mcm) * 1e6,
            cell_area_m2,
            max_attenuation_m_per_km,
        )
        attenuation_source = "solved"

    # 3. Apply the profile. The surface starts at the release level over the dam
    #    and decays with flow distance, so cells beyond its reach stay dry.
    depth = np.zeros((height, width), dtype="float32")
    local_level = np.full((height, width), -np.inf, dtype="float64")
    local_level[candidate] = (
        release_level - resolved_attenuation * distance_km[candidate]
    )
    flooded = candidate & (dem < local_level)
    depth[flooded] = np.clip((local_level - dem)[flooded], 0.0, max_depth_cap_m)

    impounded_volume_m3 = float(np.nansum(depth[flooded])) * cell_area_m2

    if not flooded.any():
        # Degenerate but survivable: the dam cell alone is always underwater.
        flooded[row, col] = True
        depth[row, col] = min(max_depth_cap_m, max(0.0, release_level - source_elevation))
        impounded_volume_m3 = float(depth[row, col]) * cell_area_m2

    return SimulationResult(
        flooded=flooded,
        depth_m=depth,
        elevation_m=dem,
        pool_level_m=release_level,
        release_level_m=release_level,
        level_source=level_source,
        source_elevation_m=source_elevation,
        release_volume_mcm=None if release_volume_mcm is None else float(release_volume_mcm),
        impounded_volume_mcm=impounded_volume_m3 / 1e6,
        volume_matched=volume_matched,
        attenuation_source=attenuation_source,
        dam_row=row,
        dam_col=col,
        dam_latitude=dam_latitude,
        dam_longitude=dam_longitude,
        cell_area_m2=cell_area_m2,
        cell_size_x_m=cell_w_m,
        cell_size_y_m=cell_h_m,
        breach_head_m=resolved_head,
        breach_width_m=resolved_width,
        attenuation_m_per_km=resolved_attenuation,
        max_depth_cap_m=float(max_depth_cap_m),
        flow_direction_east_north=(round(flow[0], 4), round(flow[1], 4)),
        snapped_to_nearest_cell=snapped,
        dam_snap_distance_km=round(snap_km, 3),
    )


def get_target_metrics(
    result: SimulationResult,
    transform,
    dem_crs,
    latitude: float,
    longitude: float,
    label: str = "",
) -> TargetMetrics:
    """Sample the simulation at a point of interest (WGS84 degrees)."""
    height, width = result.depth_m.shape
    row, col = world_to_rowcol(transform, dem_crs, latitude, longitude)

    metrics = TargetMetrics(
        latitude=float(latitude),
        longitude=float(longitude),
        label=label,
        row=row,
        col=col,
        inside_grid=0 <= row < height and 0 <= col < width,
        distance_from_dam_km=haversine_km(
            result.dam_latitude, result.dam_longitude, latitude, longitude
        ),
    )

    if not metrics.inside_grid:
        return metrics

    ground = result.elevation_m[row, col]
    if np.isfinite(ground):
        metrics.ground_elevation_m = float(ground)

    if result.flooded[row, col]:
        metrics.flood_depth_m = float(result.depth_m[row, col])
        metrics.inundated = True

    return metrics