"""Coordinate helpers and raster-mask-to-GeoJSON conversion without GDAL.

``rasterio.features.shapes`` is the usual way to get polygons out of a boolean
mask, but it drags in GDAL. This module does the same job with NumPy and a
boundary-edge tracer:

* boundary cells are found vectorised, so only the perimeter is walked,
* rings are simplified with Douglas-Peucker in metres (not in degrees, so the
  tolerance means the same thing at every latitude),
* dry islands inside the flood are preserved as interior holes,
* slivers below a minimum area are dropped,
* coordinates stay in the raster's native units — the caller reprojects if the
  DEM is not geographic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

M_PER_DEG_LAT = 110_574.0
M_PER_DEG_LON = 111_320.0

# Ring orientation produced by the tracer: exterior rings come out with negative
# signed area in a y-north coordinate system, holes with positive. Verified by
# `assert_exterior_sign()` and covered by the smoke test.
EXTERIOR_SIGN = -1


def block_mean(array: np.ndarray, factor: int) -> np.ndarray:
    """NaN-aware block average, used to bring a mosaic inside the grid budget."""
    if factor <= 1:
        return np.asarray(array, dtype="float32")

    height, width = array.shape
    out_height = max(1, math.ceil(height / factor))
    out_width = max(1, math.ceil(width / factor))

    pad_rows = out_height * factor - height
    pad_cols = out_width * factor - width
    if pad_rows or pad_cols:
        array = np.pad(
            array, ((0, pad_rows), (0, pad_cols)), constant_values=np.nan
        )

    blocks = array.reshape(out_height, factor, out_width, factor).astype(np.float64)
    valid = np.isfinite(blocks)
    counts = valid.sum(axis=(1, 3))
    totals = np.where(valid, blocks, 0.0).sum(axis=(1, 3))

    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)

    return means.astype("float32")


def is_geographic(crs) -> bool:
    """Best-effort test for a geographic (degree-based) CRS.

    Recognises the common WGS84 spellings without a GIS stack, and falls back to
    rasterio only when an unusual CRS description is supplied.
    """
    if crs is None:
        raise ValueError("The raster has no CRS information.")

    text = str(crs).upper()
    if "4326" in text or "LONGLAT" in text or "WGS 84" in text or "WGS84" in text:
        return True
    if "UTM" in text or "MERCATOR" in text or "EPSG:3857" in text:
        return False

    try:
        from rasterio.crs import CRS  # noqa: PLC0415 - optional dependency
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ValueError(
            f"Cannot tell whether CRS '{crs}' is geographic without rasterio installed."
        ) from exc

    return bool(CRS.from_user_input(crs).is_geographic)


@dataclass(slots=True)
class GridRef:
    """Affine pixel geometry plus the scaling needed for metric areas."""

    a: float
    e: float
    c: float
    f: float
    unit_to_m_x: float = 1.0
    unit_to_m_y: float = 1.0

    @classmethod
    def from_transform(cls, transform, crs, latitude: float | None = None) -> "GridRef":
        """Build from a raster transform, guessing degree-to-metre scaling."""
        if is_geographic(crs):
            reference_latitude = 0.0 if latitude is None else latitude
            scale_x = M_PER_DEG_LON * max(
                math.cos(math.radians(reference_latitude)), 1e-6
            )
            scale_y = M_PER_DEG_LAT
        else:
            scale_x = scale_y = 1.0

        return cls(
            a=float(transform.a),
            e=float(transform.e),
            c=float(transform.c),
            f=float(transform.f),
            unit_to_m_x=scale_x,
            unit_to_m_y=scale_y,
        )

    @property
    def cell_width_m(self) -> float:
        return abs(self.a) * self.unit_to_m_x

    @property
    def cell_height_m(self) -> float:
        return abs(self.e) * self.unit_to_m_y

    def lattice_to_native(self, k: float, m: float) -> tuple[float, float]:
        """Grid corner (k = column index, m = row index) to native coordinates."""
        return self.c + k * self.a, self.f + m * self.e

    def native_to_rowcol(self, x: float, y: float) -> tuple[int, int]:
        return (
            int(math.floor((y - self.f) / self.e)),
            int(math.floor((x - self.c) / self.a)),
        )


# ---------------------------------------------------------------------------
# Ring tracing
# ---------------------------------------------------------------------------
def _boundary_edges(mask: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """Directed lattice edges that separate flooded cells from dry cells.

    A vertex can carry two outgoing edges where regions pinch together, so each
    start point maps to a list of ends.
    """
    height, width = mask.shape
    padded = np.zeros((height + 2, width + 2), dtype=bool)
    padded[1:-1, 1:-1] = mask
    core = padded[1:-1, 1:-1]

    north = core & ~padded[0:-2, 1:-1]
    south = core & ~padded[2:, 1:-1]
    west = core & ~padded[1:-1, 0:-2]
    east = core & ~padded[1:-1, 2:]

    edges: dict[tuple[int, int], list[tuple[int, int]]] = {}

    def add(start: tuple[int, int], end: tuple[int, int]) -> None:
        edges.setdefault(start, []).append(end)

    for row, col in zip(*np.nonzero(north)):
        add((int(col), int(row)), (int(col) + 1, int(row)))
    for row, col in zip(*np.nonzero(east)):
        add((int(col) + 1, int(row)), (int(col) + 1, int(row) + 1))
    for row, col in zip(*np.nonzero(south)):
        add((int(col) + 1, int(row) + 1), (int(col), int(row) + 1))
    for row, col in zip(*np.nonzero(west)):
        add((int(col), int(row) + 1), (int(col), int(row)))

    return edges


def _trace_rings(
    edges: dict[tuple[int, int], list[tuple[int, int]]]
) -> list[list[tuple[int, int]]]:
    """Chain directed edges into closed rings.

    The boundary graph of a raster region is balanced (every vertex has equal in
    and out degree), so the walk always closes. ``max_steps`` is a hard stop so a
    malformed graph can never spin forever.
    """
    rings: list[list[tuple[int, int]]] = []
    max_steps = sum(len(ends) for ends in edges.values()) + 1

    for start in list(edges.keys()):
        while edges.get(start):
            ring = [start]
            current = start
            for _ in range(max_steps):
                outgoing = edges.get(current)
                if not outgoing:
                    ring = []
                    break

                following = outgoing.pop()
                ring.append(following)
                current = following
                if current == start:
                    break
            else:
                ring = []

            if len(ring) > 3:
                rings.append(ring[:-1])

    return rings


def _signed_area(ring_m: np.ndarray) -> float:
    x = ring_m[:, 0]
    y = ring_m[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _contains_point(ring_m: np.ndarray, point: tuple[float, float]) -> bool:
    """Ray casting in metric space."""
    x, y = point
    crossings = False
    count = len(ring_m)
    for index in range(count):
        x1, y1 = ring_m[index]
        x2, y2 = ring_m[(index + 1) % count]
        if (y1 > y) != (y2 > y):
            x_intersection = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_intersection:
                crossings = not crossings
    return crossings


def _simplify(ring_m: np.ndarray, tolerance_m: float) -> np.ndarray:
    """Iterative Douglas-Peucker simplification of a closed ring."""
    if tolerance_m <= 0 or len(ring_m) <= 4:
        return ring_m

    closed = np.vstack([ring_m, ring_m[:1]])
    keep = np.zeros(len(closed), dtype=bool)
    keep[0] = keep[-1] = True

    stack = [(0, len(closed) - 1)]
    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue

        segment = closed[end] - closed[start]
        length = math.hypot(segment[0], segment[1])
        points = closed[start + 1 : end]

        if length < 1e-9:
            distances = np.hypot(points[:, 0] - closed[start, 0], points[:, 1] - closed[start, 1])
        else:
            distances = np.abs(
                segment[0] * (closed[start, 1] - points[:, 1])
                - (closed[start, 0] - points[:, 0]) * segment[1]
            ) / length

        furthest = int(np.argmax(distances)) if distances.size else -1
        if furthest >= 0 and distances[furthest] > tolerance_m:
            index = start + 1 + furthest
            keep[index] = True
            stack.append((start, index))
            stack.append((index, end))

    return closed[keep][:-1]


def _ring_to_metric(ring: list[tuple[int, int]], grid: GridRef) -> np.ndarray:
    """Lattice ring -> local metric coordinates (metres)."""
    native = np.array([grid.lattice_to_native(k, m) for k, m in ring], dtype=np.float64)
    native[:, 0] *= grid.unit_to_m_x
    native[:, 1] *= grid.unit_to_m_y
    return native


def mask_to_polygons(
    mask: np.ndarray,
    grid: GridRef,
    *,
    simplify_tolerance_m: float | None = None,
    min_area_m2: float = 0.0,
) -> list[dict]:
    """Convert a boolean raster mask into GeoJSON Polygon geometries.

    All internal work happens in metres, so the simplification tolerance and the
    minimum-area filter mean the same thing at every latitude. The returned
    coordinates are native to the raster (degrees for a geographic DEM, projected
    units otherwise), with dry islands preserved as interior holes.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        return []

    if simplify_tolerance_m is None:
        simplify_tolerance_m = 0.6 * max(grid.cell_width_m, grid.cell_height_m)

    exteriors: list[np.ndarray] = []
    holes: list[np.ndarray] = []

    for ring in _trace_rings(_boundary_edges(mask)):
        metric = _ring_to_metric(ring, grid)
        area = _signed_area(metric)

        if area * EXTERIOR_SIGN <= 0:
            if len(metric) >= 3:
                holes.append(metric)
            continue

        if abs(area) < min_area_m2:
            continue

        reduced = _simplify(metric, simplify_tolerance_m)
        if len(reduced) < 3:
            reduced = metric
        if abs(_signed_area(reduced)) < min_area_m2 * 0.5:
            continue

        exteriors.append(reduced)

    polygons: list[dict] = []
    scale_x, scale_y = grid.unit_to_m_x, grid.unit_to_m_y

    for metric in exteriors:
        candidates: list[tuple[float, np.ndarray]] = []
        for hole in holes:
            point = (hole[0][0], hole[0][1])
            if not (
                metric[:, 0].min() <= point[0] <= metric[:, 0].max()
                and metric[:, 1].min() <= point[1] <= metric[:, 1].max()
            ):
                continue
            if _contains_point(metric, point):
                candidates.append((abs(_signed_area(hole)), hole))

        # Innermost first: nested islands belong to the smallest enclosing ring.
        candidates.sort(key=lambda item: item[0])

        rings_out = [_ring_to_coordinates(metric, scale_x, scale_y)]
        for _, hole in candidates:
            reduced_hole = _simplify(hole, simplify_tolerance_m)
            if len(reduced_hole) >= 3:
                rings_out.append(_ring_to_coordinates(reduced_hole, scale_x, scale_y))

        polygons.append(
            {
                "type": "Polygon",
                "coordinates": rings_out,
                "area_km2": round(abs(_signed_area(metric)) / 1e6, 4),
            }
        )

    polygons.sort(key=lambda polygon: polygon["area_km2"], reverse=True)
    return polygons


def _ring_to_coordinates(
    ring_m: np.ndarray, scale_x: float, scale_y: float, precision: int = 5
) -> list[list[float]]:
    """Metric ring back to native units, closed, and rounded for transport."""
    native = ring_m.copy()
    native[:, 0] = np.round(native[:, 0] / scale_x, precision)
    native[:, 1] = np.round(native[:, 1] / scale_y, precision)

    coordinates = native.tolist()
    if coordinates[0] != coordinates[-1]:
        coordinates.append(coordinates[0])
    return coordinates


def band_geojson(
    classes: np.ndarray,
    grid: GridRef,
    labels: tuple[str, ...] | list[str],
    *,
    bands: list[int] | None = None,
    simplify_tolerance_m: float | None = None,
    min_area_m2: float = 50_000.0,
) -> tuple[list[dict], list[float] | None]:
    """GeoJSON features for each severity band, plus an overall WGS84-free bbox.

    Features are ordered shallowest-first so the deepest zones paint on top.
    """
    features: list[dict] = []
    bbox: list[float] | None = None

    selected = bands if bands is not None else list(range(len(labels)))

    for band in sorted(selected):
        band_mask = classes == band
        if not band_mask.any():
            continue

        for polygon in mask_to_polygons(
            band_mask,
            grid,
            simplify_tolerance_m=simplify_tolerance_m,
            min_area_m2=min_area_m2,
        ):
            geometry = {"type": polygon["type"], "coordinates": polygon["coordinates"]}
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "band": band,
                        "severity": labels[band] if band < len(labels) else f"band {band}",
                        "area_km2": polygon["area_km2"],
                    },
                    "geometry": geometry,
                }
            )

            for ring in polygon["coordinates"]:
                for x, y in ring:
                    if bbox is None:
                        bbox = [x, y, x, y]
                    else:
                        bbox[0] = min(bbox[0], x)
                        bbox[1] = min(bbox[1], y)
                        bbox[2] = max(bbox[2], x)
                        bbox[3] = max(bbox[3], y)

    return features, bbox


def assert_exterior_sign() -> None:
    """Sanity check that the tracer's winding assumption still holds."""
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    grid = GridRef(a=1.0, e=-1.0, c=0.0, f=0.0)
    rings = _trace_rings(_boundary_edges(mask))
    assert len(rings) == 1, rings
    metric = np.array([grid.lattice_to_native(k, m) for k, m in rings[0]], dtype=float)
    area = _signed_area(metric)
    assert area * EXTERIOR_SIGN > 0, f"unexpected winding: area={area}"