"""Offline checks for the screening model and the polygoniser.

    python smoke_model.py

No network and no GDAL: this exercises the connected flood fill, the volumeric
level-pool solve, depth metrics, discharge estimation, target sampling, the
dam-snap report, and the mask-to-GeoJSON tracer (rings, holes, winding,
simplification, area preservation).
"""

from __future__ import annotations

import math

import numpy as np

from geometry import (
    EXTERIOR_SIGN,
    GridRef,
    assert_exterior_sign,
    band_geojson,
    mask_to_polygons,
)
from simulation import DEPTH_LABELS, get_target_metrics, simulate_breach

PASSED: list[str] = []


def check(label: str, condition: bool, detail: object = "") -> None:
    if not condition:
        raise AssertionError(f"{label} — got {detail!r}")
    PASSED.append(label)


def _signed(coordinates: list[list[float]], grid: GridRef) -> float:
    """Shoelace area of a ring in square metres."""
    points = np.array(coordinates, dtype=float)
    x = points[:, 0] * grid.unit_to_m_x
    y = points[:, 1] * grid.unit_to_m_y
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# ---------------------------------------------------------------------------
# Terrain: a ramp rising to the east, dam at column 10
# ---------------------------------------------------------------------------
HEIGHT, WIDTH = 100, 140
rows, columns = np.mgrid[0:HEIGHT, 0:WIDTH]
dem = (100 + columns * 2.0).astype("float32")

transform = GridRef(a=0.01, e=-0.01, c=76.0, f=31.5)
dam_row, dam_col = 50, 10
dam_lon = transform.c + (dam_col + 0.5) * transform.a
dam_lat = transform.f + (dam_row + 0.5) * transform.e

grid = GridRef.from_transform(transform, "EPSG:4326", dam_lat)
cell_area_m2 = grid.cell_width_m * grid.cell_height_m
cell_w = abs(transform.a) * 111_320.0 * math.cos(math.radians(dam_lat))
cell_h = abs(transform.e) * 110_574.0

# ---------------------------------------------------------------------------
# 1. Flat pool at the release level, downstream of the dam
# ---------------------------------------------------------------------------
# The ramp rises eastwards, so the dam at column 10 drains WEST. The flood is the
# reach below the dam (columns 0..11, allowing the 1.5-cell margin), not the
# rising ground behind it, which stands in for the reservoir basin.
result = simulate_breach(
    dem=dem,
    transform=transform,
    dem_crs="EPSG:4326",
    dam_latitude=dam_lat,
    dam_longitude=dam_lon,
    release_level_m=150.0,
)

expected_cells = HEIGHT * 12
check(
    "flow direction points downhill",
    result.flow_direction_east_north[0] < -0.9,
    result.flow_direction_east_north,
)
check(
    "upstream (reservoir) side stays dry",
    not result.flooded[:, 13:].any(),
    int(result.flooded[:, 13:].sum()),
)
check("flooded cell count", result.flooded_cells == expected_cells, result.flooded_cells)
check("source elevation", result.source_elevation_m == 120.0, result.source_elevation_m)
check("pool level", result.pool_level_m == 150.0, result.pool_level_m)
check("release level", result.release_level_m == 150.0, result.release_level_m)
check("no volume budget used", result.release_volume_mcm is None)
check("flat profile without a volume", result.attenuation_m_per_km == 0.0)
check("profile source recorded", result.attenuation_source == "none")
check("volume matched flag is set", result.volume_matched)
check("depth cap respected", abs(result.maximum_depth_m - 50.0) < 1e-4, result.maximum_depth_m)
check("cell width", abs(result.cell_size_x_m - cell_w) < 1e-6, result.cell_size_x_m)
check("cell height", abs(result.cell_size_y_m - cell_h) < 1e-6, result.cell_size_y_m)

expected_km2 = expected_cells * cell_area_m2 / 1e6
check(
    "flooded area km2",
    abs(result.estimated_flooded_area_km2 - expected_km2) < 1e-9,
    result.estimated_flooded_area_km2,
)

mean_depth = 150.0 - (100.0 + 122.0) / 2.0
check(
    "flood volume",
    abs(result.flood_volume_m3 - mean_depth * expected_cells * cell_area_m2) < 1.0,
    result.flood_volume_m3,
)

# Breach head defaults to the release level above the dam terrain (30 m).
check("default breach head", abs(result.breach_head_m - 30.0) < 1e-9, result.breach_head_m)
expected_discharge = (8.0 / 27.0) * 30.0 * math.sqrt(9.80665) * 30.0**1.5
check(
    "peak discharge",
    abs(result.peak_discharge_m3s - expected_discharge) < 1e-6,
    result.peak_discharge_m3s,
)
check("breach width defaults to head", result.breach_width_m == 30.0, result.breach_width_m)

# ---------------------------------------------------------------------------
# 2. Volume-solved surface gradient (the headline behaviour)
# ---------------------------------------------------------------------------
flat_volume_m3 = result.flood_volume_m3
half_budget_m3 = flat_volume_m3 / 2.0

solved = simulate_breach(
    dem=dem,
    transform=transform,
    dem_crs="EPSG:4326",
    dam_latitude=dam_lat,
    dam_longitude=dam_lon,
    release_level_m=150.0,
    release_volume_mcm=half_budget_m3 / 1e6,
)
check("gradient was solved", solved.attenuation_source == "solved")
check("solved gradient is positive", solved.attenuation_m_per_km > 0, solved.attenuation_m_per_km)
check("solved volume matches the budget", solved.volume_matched)
check(
    "impounded volume equals the budget",
    abs(solved.flood_volume_m3 - half_budget_m3) / half_budget_m3 < 2e-3,
    (solved.flood_volume_m3, half_budget_m3),
)
check(
    "pool still starts at the release level",
    solved.pool_level_m == 150.0 and solved.release_level_m == 150.0,
    solved.pool_level_m,
)
check(
    "half the water floods less ground",
    solved.flooded_cells < result.flooded_cells,
    (solved.flooded_cells, result.flooded_cells),
)
check(
    "depth decays away from the dam",
    solved.depth_m[dam_row, dam_col] > solved.depth_m[dam_row, dam_col + 10],
    (solved.depth_m[dam_row, dam_col], solved.depth_m[dam_row, dam_col + 10]),
)
# The gradient can only lower the surface, never raise it.
check(
    "gradient never deepens the flood",
    solved.maximum_depth_m <= result.maximum_depth_m + 1e-6,
    (solved.maximum_depth_m, result.maximum_depth_m),
)

# A budget larger than the window can hold must be reported, not invented away.
huge = simulate_breach(
    dem=dem,
    transform=transform,
    dem_crs="EPSG:4326",
    dam_latitude=dam_lat,
    dam_longitude=dam_lon,
    release_level_m=150.0,
    release_volume_mcm=flat_volume_m3 * 10 / 1e6,
)
check("oversized budget keeps the pool flat", huge.attenuation_m_per_km == 0.0)
check("oversized budget is flagged", not huge.volume_matched)
check(
    "oversized budget floods the whole flat pool",
    huge.flooded_cells == expected_cells,
    huge.flooded_cells,
)
check(
    "window capacity reported",
    abs(huge.impounded_volume_mcm * 1e6 - flat_volume_m3) / flat_volume_m3 < 1e-4,
    huge.impounded_volume_mcm,
)

# A manual gradient is respected and reported as such.
manual = simulate_breach(
    dem=dem,
    transform=transform,
    dem_crs="EPSG:4326",
    dam_latitude=dam_lat,
    dam_longitude=dam_lon,
    release_level_m=150.0,
    release_volume_mcm=half_budget_m3 / 1e6,
    attenuation_m_per_km=0.5,
)
check("manual gradient is used", manual.attenuation_m_per_km == 0.5)
check("manual gradient is labelled", manual.attenuation_source == "user")
check(
    "manual gradient overrides the solve",
    abs(manual.flood_volume_m3 - half_budget_m3) / half_budget_m3 > 1e-3,
    manual.flood_volume_m3,
)


# ---------------------------------------------------------------------------
# 3. Connectivity, attenuation, bands, head-only fallback
# ---------------------------------------------------------------------------
# A low basin behind an impassable ridge on the downstream side must stay dry.
pit = dem.copy()
pit[:, 2] = 400.0
pit[:, 0:2] = 5.0
pit_result = simulate_breach(
    dem=pit,
    transform=transform,
    dem_crs="EPSG:4326",
    dam_latitude=dam_lat,
    dam_longitude=dam_lon,
    release_level_m=150.0,
)
check("disconnected basin stays dry", not pit_result.flooded[:, 0:2].any())

attenuated = simulate_breach(
    dem=dem,
    transform=transform,
    dem_crs="EPSG:4326",
    dam_latitude=dam_lat,
    dam_longitude=dam_lon,
    release_level_m=150.0,
    attenuation_m_per_km=0.5,
)
check(
    "attenuation shrinks the footprint",
    attenuated.flooded_cells < result.flooded_cells,
    (attenuated.flooded_cells, result.flooded_cells),
)

head_only = simulate_breach(
    dem=dem,
    transform=transform,
    dem_crs="EPSG:4326",
    dam_latitude=dam_lat,
    dam_longitude=dam_lon,
    breach_head_m=30.0,
)
check("head-only fallback matches the level run", head_only.flooded_cells == expected_cells)
check("head-only source label", head_only.level_source == "dam terrain + breach head")

try:
    simulate_breach(
        dem=dem,
        transform=transform,
        dem_crs="EPSG:4326",
        dam_latitude=dam_lat,
        dam_longitude=dam_lon,
        release_level_m=100.0,  # below the dam terrain
    )
    check("level below terrain is rejected", False)
except ValueError:
    check("level below terrain is rejected", True)

classes = result.depth_class_grid()
counts = [int((classes == band).sum()) for band in range(len(DEPTH_LABELS))]
check("band counts sum to flood", sum(counts) == expected_cells, counts)
check("deepest band at the far edge", classes[0, 0] == 4, int(classes[0, 0]))
check("no shallow band on this ramp", counts[0] == 0, counts)

# ---------------------------------------------------------------------------
# 4. Target sampling and snapping
# ---------------------------------------------------------------------------
inside_lon = transform.c + 5.5 * transform.a
outside_lon = transform.c + 60.5 * transform.a

inside = get_target_metrics(result, transform, "EPSG:4326", dam_lat, inside_lon, "Area 1")
outside = get_target_metrics(result, transform, "EPSG:4326", dam_lat, outside_lon, "Area 2")

check("target inside floods", inside.inundated, inside.as_dict())
check("target depth", abs(inside.flood_depth_m - 40.0) < 1e-4, inside.flood_depth_m)
check("target ground elevation", inside.ground_elevation_m == 110.0, inside.ground_elevation_m)
check("target severity", inside.severity == "extreme", inside.severity)
check(
    "target distance from dam",
    abs(inside.distance_from_dam_km - 4.77) < 0.1,
    inside.distance_from_dam_km,
)
check("dry target", not outside.inundated and outside.severity == "dry", outside.as_dict())

far = get_target_metrics(result, transform, "EPSG:4326", dam_lat, 200.0, "Far")
check("outside grid is reported", not far.inside_grid and not far.inundated)

# The dam is requested far east of the grid, so it snaps to the eastern edge
# (378 m elevation there), which is why the release level has to clear it.
snapped = simulate_breach(
    dem=dem,
    transform=transform,
    dem_crs="EPSG:4326",
    dam_latitude=dam_lat,
    dam_longitude=transform.c + 40.0,
    release_level_m=420.0,
)
check("snap detected", snapped.snapped_to_nearest_cell)
check("snap distance reported", snapped.dam_snap_distance_km > 10, snapped.dam_snap_distance_km)

# ---------------------------------------------------------------------------
# 5. Polygoniser
# ---------------------------------------------------------------------------
assert_exterior_sign()
check("winding assumption holds", EXTERIOR_SIGN == -1)

mask = np.ones((20, 20), dtype=bool)
mask[7:13, 7:13] = False  # dry island in the middle

polygons = mask_to_polygons(mask, grid, simplify_tolerance_m=0.0, min_area_m2=0.0)
check("one polygon traced", len(polygons) == 1, len(polygons))
check(
    "island preserved as a hole",
    len(polygons[0]["coordinates"]) == 2,
    polygons[0]["coordinates"],
)

exterior_area = abs(_signed(polygons[0]["coordinates"][0], grid))
island_area = abs(_signed(polygons[0]["coordinates"][1], grid))
check(
    "exterior area equals 400 cells",
    abs(exterior_area - 400 * cell_area_m2) / (400 * cell_area_m2) < 0.01,
    exterior_area,
)
check(
    "island area equals 36 cells",
    abs(island_area - 36 * cell_area_m2) / (36 * cell_area_m2) < 0.01,
    island_area,
)

check("fully flooded mask still yields a ring", len(mask_to_polygons(np.ones((8, 8), bool), grid)) == 1)
check("empty mask yields nothing", mask_to_polygons(np.zeros((8, 8), bool), grid) == [])
check(
    "minimum area filter drops slivers",
    mask_to_polygons(np.ones((2, 2), bool), grid, min_area_m2=1e12) == [],
)

steps = np.zeros((30, 30), dtype=bool)
for index in range(20):
    steps[index, : 10 + (index // 2)] = True
raw_ring = len(
    mask_to_polygons(steps, grid, simplify_tolerance_m=0.0, min_area_m2=0.0)[0]["coordinates"][0]
)
simplified_ring = len(
    mask_to_polygons(steps, grid, simplify_tolerance_m=200.0, min_area_m2=0.0)[0]["coordinates"][0]
)
check("simplification removes vertices", simplified_ring < raw_ring, (raw_ring, simplified_ring))

features, bbox = band_geojson(classes, grid, DEPTH_LABELS, min_area_m2=0.0)
bands = [feature["properties"]["band"] for feature in features]
check("features emitted", len(features) > 0, len(features))
check("features ordered shallowest first", bands == sorted(bands), bands)
check("bbox reported", bbox is not None and bbox[2] > bbox[0], bbox)

# ---------------------------------------------------------------------------
# 6. Plots: profile, cross-section and the severity split
# ---------------------------------------------------------------------------
# These need the web stack (main.py imports fastapi and pandas) on top of the
# numpy the model checks above use — everything else in this file runs offline.
from main import _band_areas, _cross_section, _longitudinal_profile  # noqa: E402

# A valley floor that climbs again downstream and a ridge across the flow, so
# the flood has a real edge to draw and the cross-section has real banks to
# widen onto.
plot_rows, plot_cols = np.mgrid[0:40, 0:90]
plot_dem = (
    100.0
    + (plot_cols - 30.0) ** 2 / 100.0
    - plot_cols * 0.3
    + 0.5 * (plot_rows - 20.0) ** 2
).astype("float32")
plot_transform = GridRef(a=0.01, e=-0.01, c=76.0, f=31.5)
plot_lon = plot_transform.c + 60.5 * plot_transform.a
plot_lat = plot_transform.f + 20.5 * plot_transform.e
plot_grid = GridRef.from_transform(plot_transform, "EPSG:4326", plot_lat)

plot_result = simulate_breach(
    dem=plot_dem,
    transform=plot_transform,
    dem_crs="EPSG:4326",
    dam_latitude=plot_lat,
    dam_longitude=plot_lon,
    release_level_m=100.0,
)
check("plot fixture floods", plot_result.flooded_cells > 0, plot_result.flooded_cells)

band_areas = _band_areas(plot_result)
check("one label per severity band", len(band_areas["labels"]) == len(DEPTH_LABELS))
check(
    "band areas split the headline total",
    abs(sum(band_areas["area_km2"]) - plot_result.estimated_flooded_area_km2) < 1e-3,
    (sum(band_areas["area_km2"]), plot_result.estimated_flooded_area_km2),
)
check(
    "every band area is non-negative",
    all(value >= 0 for value in band_areas["area_km2"]),
    band_areas["area_km2"],
)

SAMPLES = 121
profile = _longitudinal_profile(plot_result, plot_grid, geographic=True, samples=SAMPLES)
check("profile sample count", len(profile["distance_km"]) == SAMPLES, len(profile["distance_km"]))
check("profile starts at the dam", profile["distance_km"][0] == 0.0, profile["distance_km"][0])
check("profile reports its reach", profile["reach_km"] > 0, profile["reach_km"])
check("profile carries coordinates", profile["latitude"] is not None)
check(
    "profile surface starts at the pool",
    abs(profile["water_surface_m"][0] - plot_result.pool_level_m) < 1e-6,
    (profile["water_surface_m"][0], plot_result.pool_level_m),
)
check(
    "profile surface never rises downstream",
    all(
        later <= earlier + 1e-6
        for earlier, later in zip(
            profile["water_surface_m"], profile["water_surface_m"][1:]
        )
    ),
)
check(
    "profile terrain is continuous",
    all(value is not None for value in profile["terrain_m"]),
    sum(value is None for value in profile["terrain_m"]),
)
wet_indices = [i for i, flag in enumerate(profile["flooded"]) if flag]
check("profile reaches wet ground", bool(wet_indices), len(wet_indices))
check("profile reaches dry ground", len(wet_indices) < len(profile["flooded"]), len(wet_indices))
last_wet = wet_indices[-1]
check(
    "ground is under the surface while flooded",
    profile["terrain_m"][last_wet] < profile["water_surface_m"][last_wet],
    (profile["terrain_m"][last_wet], profile["water_surface_m"][last_wet]),
)
check(
    "ground breaks the surface once flooding stops",
    profile["terrain_m"][last_wet + 1] >= profile["water_surface_m"][last_wet + 1],
    (profile["terrain_m"][last_wet + 1], profile["water_surface_m"][last_wet + 1]),
)
check("profile depth is zero when dry", profile["depth_m"][-1] == 0.0, profile["depth_m"][-1])

section = _cross_section(
    plot_result,
    plot_transform,
    "EPSG:4326",
    latitude=plot_lat,
    longitude=plot_lon,
    label="Area 1",
    samples=SAMPLES,
)
check("section emitted", section is not None)
check("section sample count", len(section["offset_km"]) == SAMPLES, len(section["offset_km"]))
if section:
    check(
        "section offsets straddle the centre",
        section["offset_km"][0] < 0 < section["offset_km"][-1],
        section["offset_km"],
    )
    check("section labels itself", section["label"] == "Area 1", section["label"])
    check("section reports the flood width", section["flood_width_km"] > 0, section["flood_width_km"])
    check(
        "section widened onto dry ground",
        section["flood_width_km"] < section["span_km"],
        (section["flood_width_km"], section["span_km"]),
    )
    check("section is dry at both ends", not section["flooded"][0] and not section["flooded"][-1])
    dry_end, wet_end = section["flooded"][0], section["flooded"][len(section["flooded"]) // 2]
    check("section centre is in the water", wet_end, (dry_end, wet_end))

print(f"ALL MODEL CHECKS PASSED ({len(PASSED)} assertions)")
print(
    {
        "flooded_km2": round(result.estimated_flooded_area_km2, 2),
        "cells": result.flooded_cells,
        "max_depth_m": round(result.maximum_depth_m, 2),
        "volume_mcm": round(result.flood_volume_m3 / 1e6, 3),
        "peak_discharge_m3s": round(result.peak_discharge_m3s, 1),
        "solved_pool_m": solved.pool_level_m,
        "zone_features": len(features),
    }
)