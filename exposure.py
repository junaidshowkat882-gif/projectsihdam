"""Exposure lookup: which mapped assets sit inside the modelled flood?

OpenStreetMap's Overpass API is free, key-less and queryable by bounding box, so
the screening result can be turned into something a district officer actually
needs: *how many schools, hospitals and power assets are inside the water, and
how deep is it there?*

Everything here degrades quietly. Overpass is a shared community service: it can
be slow, rate-limited or offline, so any failure returns ``None`` and the
simulation response simply omits the exposure block.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from simulation import SimulationResult, world_to_rowcol

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
REQUEST_TIMEOUT_S = 30
USER_AGENT = "SIH26161-flood-screening/1.2 (screening prototype)"
CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "overpass"

MAX_ASSETS_RETURNED = 250
MAX_EXAMPLES = 3


@dataclass(frozen=True, slots=True)
class AssetGroup:
    """One class of mapped asset worth naming separately in a briefing."""

    key: str
    label: str
    selector: str
    # Place names live on nodes; asking for matching *ways* too would just add
    # payload and latency without finding anything.
    query_ways: bool = True


GROUPS: tuple[AssetGroup, ...] = (
    AssetGroup(
        "education",
        "Schools & colleges",
        '["amenity"~"^(school|college|kindergarten|university)$"]',
    ),
    AssetGroup(
        "health",
        "Hospitals & clinics",
        '["amenity"~"^(hospital|clinic|doctors)$"]',
    ),
    AssetGroup(
        "power",
        "Power infrastructure",
        '["power"~"^(plant|substation)$"]',
    ),
    AssetGroup(
        "water",
        "Water & sanitation",
        '["man_made"~"^(water_works|water_tower|wastewater_plant)$"]',
    ),
    AssetGroup(
        "settlement",
        "Settlements",
        '["place"~"^(city|town|village|hamlet)$"]',
        query_ways=False,
    ),
    AssetGroup(
        "transport",
        "Transport nodes",
        '["railway"~"^(station|halt)$"]',
        query_ways=False,
    ),
)


# The cached Overpass payload only contains the tags the *current* group list
# asked for, so the group definitions have to be part of the cache key. Without
# this, a response cached before a group was added keeps being served and the new
# category silently reports zero instead of being re-queried.
GROUPS_FINGERPRINT = hashlib.sha1(
    "|".join(
        f"{group.key}:{group.selector}:{group.query_ways}" for group in GROUPS
    ).encode("utf-8")
).hexdigest()[:10]


def _cache_path(south: float, west: float, north: float, east: float) -> Path:
    key = f"{south:.3f}_{west:.3f}_{north:.3f}_{east:.3f}"
    return CACHE_DIR / f"{key}_{GROUPS_FINGERPRINT}.json"


def fetch_assets(
    south: float,
    west: float,
    north: float,
    east: float,
    *,
    use_cache: bool = True,
) -> list[dict] | None:
    """Return raw OSM elements (with centres) inside the bounding box."""
    if not (south < north and west < east):
        return None

    cache_path = _cache_path(south, west, north, east)
    if use_cache and cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cache_path.unlink(missing_ok=True)

    bbox = f"{south:.5f},{west:.5f},{north:.5f},{east:.5f}"
    selectors = [f"node{group.selector}({bbox});" for group in GROUPS] + [
        f"way{group.selector}({bbox});" for group in GROUPS if group.query_ways
    ]
    query = f"[out:json][timeout:25];\n(\n  " + "\n  ".join(selectors) + "\n);\nout center tags;"

    payload = urllib.parse.urlencode({"data": query}).encode("utf-8")
    request = urllib.request.Request(
        OVERPASS_URL,
        data=payload,
        headers={"User-Agent": USER_AGENT, "Content-Type": "application/x-www-form-urlencoded"},
    )

    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None

    elements = data.get("elements", [])
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            cache_path.write_text(json.dumps(elements), encoding="utf-8")
        except OSError:
            pass

    return elements


def _element_position(element: dict) -> tuple[float, float] | None:
    if element.get("type") == "node":
        latitude, longitude = element.get("lat"), element.get("lon")
    else:
        centre = element.get("center") or {}
        latitude, longitude = centre.get("lat"), centre.get("lon")

    if latitude is None or longitude is None:
        return None
    return float(latitude), float(longitude)


# Tag values that map onto each exposure group (kept apart from GROUPS so the
# Overpass selector and the classifier cannot drift).
_EDUCATION = {"school", "college", "kindergarten", "university"}
_HEALTH = {"hospital", "clinic", "doctors"}
_POWER = {"plant", "substation"}
_WATER = {"water_works", "water_tower", "wastewater_plant"}
_SETTLEMENT = {"city", "town", "village", "hamlet"}
_TRANSPORT = {"station", "halt"}

_GROUPS_BY_KEY = {group.key: group for group in GROUPS}


def _group_for(tags: dict) -> AssetGroup | None:
    """Classify an OSM element into one of the exposure groups."""
    amenity = tags.get("amenity")
    if amenity in _EDUCATION:
        return _GROUPS_BY_KEY["education"]
    if amenity in _HEALTH:
        return _GROUPS_BY_KEY["health"]
    if tags.get("power") in _POWER:
        return _GROUPS_BY_KEY["power"]
    if tags.get("man_made") in _WATER:
        return _GROUPS_BY_KEY["water"]
    if tags.get("place") in _SETTLEMENT:
        return _GROUPS_BY_KEY["settlement"]
    if tags.get("railway") in _TRANSPORT:
        return _GROUPS_BY_KEY["transport"]
    return None


def _asset_name(tags: dict, group: AssetGroup, identifier: str) -> str:
    for key in ("name:en", "name", "operator"):
        value = tags.get(key)
        if value:
            return str(value)[:70]
    return f"Unnamed {group.label.lower()} (OSM {identifier})"


def analyse_exposure(
    result: SimulationResult,
    transform,
    dem_crs,
    bounds_wgs84: tuple[float, float, float, float],
    *,
    use_cache: bool = True,
) -> dict | None:
    """Cross-reference mapped assets with the flood mask.

    Args:
        bounds_wgs84: ``(west, south, east, north)`` search window in degrees.

    Returns:
        A summary dict, or ``None`` when Overpass is unavailable so callers can
        simply omit the block.
    """
    west, south, east, north = bounds_wgs84

    elements = fetch_assets(south, west, north, east, use_cache=use_cache)
    if elements is None:
        return None

    height, width = result.depth_m.shape
    assets: list[dict] = []
    tallies: dict[str, dict] = {
        group.key: {"total": 0, "inundated": 0, "max_depth_m": 0.0, "examples": []}
        for group in GROUPS
    }

    for element in elements:
        tags = element.get("tags") or {}
        group = _group_for(tags)
        if group is None:
            continue

        position = _element_position(element)
        if position is None:
            continue

        latitude, longitude = position
        tally = tallies[group.key]
        tally["total"] += 1

        row, col = world_to_rowcol(transform, dem_crs, latitude, longitude)
        depth = 0.0
        inundated = False

        if 0 <= row < height and 0 <= col < width and result.flooded[row, col]:
            inundated = True
            depth = float(result.depth_m[row, col])

        name = _asset_name(tags, group, element.get("id", "?"))

        if inundated:
            tally["inundated"] += 1
            tally["max_depth_m"] = max(tally["max_depth_m"], depth)
            if len(tally["examples"]) < MAX_EXAMPLES:
                tally["examples"].append(name)

        assets.append(
            {
                "id": element.get("id"),
                "group": group.key,
                "group_label": group.label,
                "name": name,
                "latitude": round(latitude, 6),
                "longitude": round(longitude, 6),
                "inundated": inundated,
                "flood_depth_m": round(depth, 2),
            }
        )

    assets.sort(key=lambda asset: (not asset["inundated"], -asset["flood_depth_m"]))

    groups = [
        {
            "key": group.key,
            "label": group.label,
            "total": tallies[group.key]["total"],
            "inundated": tallies[group.key]["inundated"],
            "max_depth_m": round(tallies[group.key]["max_depth_m"], 2),
            "examples": tallies[group.key]["examples"],
        }
        for group in GROUPS
    ]

    return {
        "available": True,
        "source": "OpenStreetMap via Overpass API",
        "licence": "© OpenStreetMap contributors (ODbL)",
        "bbox_wgs84": [west, south, east, north],
        "groups": groups,
        "assets_total": len(assets),
        "assets_inundated": sum(1 for asset in assets if asset["inundated"]),
        "assets": assets[:MAX_ASSETS_RETURNED],
        "truncated": len(assets) > MAX_ASSETS_RETURNED,
    }


def clear_cache() -> int:
    """Delete cached Overpass responses; returns the number of files removed."""
    if not CACHE_DIR.exists():
        return 0
    removed = 0
    for path in CACHE_DIR.glob("*.json"):
        path.unlink(missing_ok=True)
        removed += 1
    return removed