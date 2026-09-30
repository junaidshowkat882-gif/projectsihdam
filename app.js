/* ==========================================================================
   SIH26161 · Dam-Breach Flood Screening — frontend controller
   --------------------------------------------------------------------------
   Talks to the FastAPI backend (automatic global terrain, real screening model)
   and falls back to a deterministic synthetic footprint when no API is
   reachable, so the interface stays explorable offline.
   ========================================================================== */

(() => {
  "use strict";

  const DEFAULT_API = location.protocol.startsWith("http")
    ? location.origin
    : "http://127.0.0.1:8010";

  const FALLBACK = {
    apiUrl: DEFAULT_API,
    probeTimeoutMs: 3000,
    breachHeadM: 30,
    radiusKm: 30,
    defaultRadiusKm: 30,
    headEdited: false,
    levelEdited: false,
    volumeEdited: false,
    maxRadiusKm: 40,
    maxUploadMb: 60,
    mapCenter: [22.5, 79.0],
    mapZoom: 5,
  };


  const BAND_COLORS = ["#22d3ee", "#38bdf8", "#fbbf24", "#fb923c", "#f43f5e"];
  const BAND_LABELS = ["<0.5 m", "0.5–1 m", "1–2 m", "2–5 m", ">5 m"];
  const SNAP_WARN_KM = 5;

  const SEVERITY_BADGE = {
    dry: "badge--dry",
    minor: "badge--minor",
    moderate: "badge--moderate",
    severe: "badge--severe",
    extreme: "badge--extreme",
  };

  /* ============================================================ 3-D valley
     Real Three.js scene over the *modelled* grids — the downsampled DEM plus
     the solved depth field from payload.grid3d. Terrain mesh (hypsometric
     tint + hillshade), water surface sitting on ground + depth, dam wall at
     the dam cell, arrival-time flood reveal driven by the same front curve as
     the 2-D replay. Dam controls (breach opening / water level / vertical
     lift) rescale the *rendered* meshes only — every number on screen stays
     the payload's, and a caption says so.
     ========================================================================= */

  const VALLEY = {
    open: false,
    three: null, // loaded Three.js module (lazy, cached)
    loading: null, // in-flight import() promise
    scene: null,
    camera: null,
    renderer: null,
    canvas: null,
    terrain: null,
    water: null,
    dam: null,
    targets: [],
    rays: null, // arrival-time contour rings
    grid: null, // decoded payload.grid3d
    damInfo: null, // dam name/state/height + payload water level
    breach: 1, // 0..1 fraction of the solved depth field shown
    level: 1, // 0..1 water-surface lift between ground and full depth
    lift: 1.8, // vertical exaggeration
    clockS: 0,
    playing: true,
    spin: false,
    lastMs: 0,
    raf: 0,
    hooks: false,
    orbit: { theta: 0.85, phi: 1.02, radius: 1.6, target: [0, 0, 0] },
    size: { w: 0, h: 0 },
    hudKey: "",
  };

  const VALLEY_CDN = "https://unpkg.com/three@0.160.0/build/three.module.js";

  // SIM_HORIZON/SIM_RUN_SECONDS live further down next to the 2-D replay, but
  // the valley front curve needs them at call time (not parse time), so there
  // is no ordering hazard: both are const in the same closure.

  // Depth tint ramp shared with the 2-D bands: shallow cyan -> red abyss.
  const VALLEY_RAMP = [
    [0.0, [34, 211, 238]],
    [0.04, [56, 189, 248]],
    [0.12, [251, 191, 36]],
    [0.35, [251, 146, 60]],
    [1.0, [244, 63, 94]],
  ];

  function valleyRamp(depthM, maxDepthM) {
    const t = clamp(maxDepthM > 0 ? depthM / maxDepthM : 0, 0, 1);
    for (let stop = 1; stop < VALLEY_RAMP.length; stop += 1) {
      if (t <= VALLEY_RAMP[stop][0]) {
        const [t0, c0] = VALLEY_RAMP[stop - 1];
        const [t1, c1] = VALLEY_RAMP[stop];
        const mix = (t - t0) / Math.max(t1 - t0, 1e-6);
        return [
          Math.round(c0[0] + (c1[0] - c0[0]) * mix),
          Math.round(c0[1] + (c1[1] - c0[1]) * mix),
          Math.round(c0[2] + (c1[2] - c0[2]) * mix),
        ];
      }
    }
    return VALLEY_RAMP[VALLEY_RAMP.length - 1][1];
  }

  const BASEMAPS = {
    // CARTO's public CDN now answers unauthenticated requests with an
    // "API KEY REQUIRED" placeholder tile rather than geography, so the dark
    // basemap uses Esri's key-less World Dark Gray instead — verified dark
    // (mean RGB ~70) and covering z0–16 with no account.
    dark: {
      url:
        "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/" +
        "World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
      attribution: "Basemap © Esri, HERE, Garmin, © OpenStreetMap contributors",
      maxZoom: 19,
      maxNativeZoom: 16,
    },
    terrain: {
      url: "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
      attribution: "© OpenTopoMap (CC-BY-SA), © OpenStreetMap contributors",
      maxZoom: 17,
      maxNativeZoom: 17,
    },
    satellite: {
      url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      attribution: "Imagery © Esri, Maxar, Earthstar Geographics",
      maxZoom: 18,
      maxNativeZoom: 18,
    },
  };

  /* ------------------------------------------------------------- utilities */
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

  const fmt = (value, decimals = 2) =>
    Number.isFinite(Number(value)) ? Number(value).toFixed(decimals) : "—";

  const group = (value, decimals = 0) =>
    Number.isFinite(Number(value))
      ? Number(value).toLocaleString("en-IN", {
          minimumFractionDigits: decimals,
          maximumFractionDigits: decimals,
        })
      : "—";

  const esc = (value) =>
    String(value ?? "").replace(
      /[&<>"']/g,
      (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]
    );

  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

  function hashString(text) {
    let hash = 2166136261;
    for (let index = 0; index < text.length; index += 1) {
      hash ^= text.charCodeAt(index);
      hash = Math.imul(hash, 16777619);
    }
    return hash >>> 0;
  }

  function mulberry32(seed) {
    let state = seed >>> 0;
    return () => {
      state = (state + 0x6d2b79f5) | 0;
      let t = Math.imul(state ^ (state >>> 15), 1 | state);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function haversineKm(lat1, lon1, lat2, lon2) {
    const radius = 6371.0088;
    const toRad = Math.PI / 180;
    const dLat = (lat2 - lat1) * toRad;
    const dLon = (lon2 - lon1) * toRad;
    const a =
      Math.sin(dLat / 2) ** 2 +
      Math.cos(lat1 * toRad) * Math.cos(lat2 * toRad) * Math.sin(dLon / 2) ** 2;
    return 2 * radius * Math.asin(Math.min(1, Math.sqrt(a)));
  }

  function toast(message, kind = "info", timeout = 4500) {
    const stack = $("#toasts");
    const element = document.createElement("div");
    element.className = `toast toast--${kind}`;
    element.textContent = message;
    stack.appendChild(element);
    setTimeout(() => {
      element.classList.add("is-leaving");
      setTimeout(() => element.remove(), 220);
    }, timeout);
  }

  function paintRange(input) {
    const min = Number(input.min || 0);
    const max = Number(input.max || 100);
    paintFill(input, ((Number(input.value) - min) / (max - min)) * 100);
  }

  function paintFill(input, percent) {
    input.style.setProperty("--fill", `${percent}%`);
  }

  function animateNumber(element, target, decimals) {
    const start = performance.now();
    const duration = 520;
    const step = (now) => {
      const t = clamp((now - start) / duration, 0, 1);
      const eased = 1 - (1 - t) ** 3;
      element.textContent = (target * eased).toFixed(decimals);
      if (t < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  /* ----------------------------------------------------------------- state */
  const state = {
    mode: "unknown",
    apiUrl: new URLSearchParams(location.search).get("api") ||
      localStorage.getItem("flood.apiUrl") ||
      FALLBACK.apiUrl,
    dams: [],
    dam: null,
    terrainSource: "auto",
    demFile: null,
    widthEdited: false,
    pickTarget: null,
    payload: null,
    busy: false,
    labels: BAND_LABELS.slice(),
    limits: {
      maxRadiusKm: FALLBACK.maxRadiusKm,
      maxUploadMb: FALLBACK.maxUploadMb,
      uploadAvailable: false,
    },
    provider: "AWS Terrain Tiles (Mapzen Terrarium)",
    uploadReason: "",
  };

  let map = null;
  let tileLayer = null;
  let damMarker = null;
  let floodLayer = null;
  let assetLayer = null;
  let tileFailures = 0;
  const targetMarkers = { 1: null, 2: null };

  /* ------------------------------------------------------------------- map */
  const mapReady = () => map !== null;

  function initMap() {
    if (typeof L === "undefined") {
      $(".stage__top").hidden = true;
      $(".stage__bottom").hidden = true;
      $("#legend").hidden = true;
      $("#map").hidden = true;
      $("#map-fallback").hidden = false;
      return;
    }

    map = L.map("map", {
      center: FALLBACK.mapCenter,
      zoom: FALLBACK.mapZoom,
      worldCopyJump: true,
      preferCanvas: true,
    });

    setBasemap("dark");

    // Metric scale bar — a flood map that can't be read for distance is a
    // picture, not a plot.
    L.control.scale({ position: "bottomleft", metric: true, imperial: false }).addTo(map);

    map.on("tileerror", () => {
      tileFailures += 1;
      if (tileFailures === 6) toast("Basemap tiles are unreachable — check the connection.", "warn");
    });

    map.on("mousemove", (event) => {
      $("#cursor-readout").textContent =
        `${event.latlng.lat.toFixed(5)}, ${event.latlng.lng.toFixed(5)}`;
    });

    map.on("click", (event) => {
      if (!state.pickTarget) return;
      const index = state.pickTarget;
      const { lat, lng } = event.latlng;
      $(`#area${index}-lat`).value = lat.toFixed(5);
      $(`#area${index}-lon`).value = lng.toFixed(5);
      setPickMode(null);
      updateTargetMarker(index);
      toast(`Area ${index} set to ${lat.toFixed(4)}, ${lng.toFixed(4)}`, "success");
    });
  }

  function setBasemap(name) {
    if (!mapReady()) return;
    const config = BASEMAPS[name] || BASEMAPS.dark;
    if (tileLayer) map.removeLayer(tileLayer);
    tileLayer = L.tileLayer(config.url, {
      attribution: config.attribution,
      maxZoom: config.maxZoom,
      // Lets Leaflet serve the provider's top zoom stretched rather than
      // leaving blank squares when the user zooms past it.
      maxNativeZoom: config.maxNativeZoom,
    }).addTo(map);
    // A basemap the user just switched to should not inherit the previous
    // layer's failure tally.
    tileFailures = 0;
    $$(".segmented__btn").forEach((button) =>
      button.classList.toggle("is-active", button.dataset.basemap === name)
    );
  }

  function popup(title, rows) {
    const body = rows
      .map(([label, value]) => `<div class="popup__row"><span>${esc(label)}</span><span>${esc(value)}</span></div>`)
      .join("");
    return `<div class="popup__title">${esc(title)}</div>${body}`;
  }

  function updateDamMarker() {
    if (!mapReady() || !state.dam) return;
    const dam = state.dam;
    if (damMarker) map.removeLayer(damMarker);
    damMarker = L.marker([dam.latitude, dam.longitude], { riseOnHover: true })
      .addTo(map)
      .bindPopup(
        popup(dam.name, [
          ["State", dam.state],
          ["Reservoir level", dam.reservoir_level_m ? `${dam.reservoir_level_m} m MSL` : "—"],
          ["Gross storage", dam.storage_mcm ? `${group(dam.storage_mcm)} MCM` : "—"],
          ["Dam height", dam.dam_height_m ? `${dam.dam_height_m} m` : "—"],
        ])
      );
  }

  function updateTargetMarker(index) {
    if (!mapReady()) return;
    const lat = Number($(`#area${index}-lat`).value);
    const lon = Number($(`#area${index}-lon`).value);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;

    if (targetMarkers[index]) map.removeLayer(targetMarkers[index]);
    targetMarkers[index] = L.circleMarker([lat, lon], {
      radius: 7,
      color: "#e6f0f9",
      weight: 2,
      fillColor: "#0a111c",
      fillOpacity: 1,
      dashArray: "3 3",
    })
      .addTo(map)
      .bindTooltip(`Area ${index}`, { direction: "top", offset: [0, -8] });
  }

  function setPickMode(index) {
    state.pickTarget = index;
    $$(".btn--pick").forEach((button) => {
      const active = Number(button.dataset.pick) === index;
      button.setAttribute("aria-pressed", String(active));
      button.textContent = active ? "Click map…" : "Pick on map";
    });
    if (mapReady()) map.getContainer().style.cursor = index ? "crosshair" : "";
  }

  function renderLegend() {
    const list = $("#legend-list");
    if (!list) return;
    list.innerHTML = state.labels
      .map((label, band) => ({ label, band }))
      .reverse()
      .map(
        ({ label, band }) =>
          `<li><i style="--swatch: ${BAND_COLORS[band] ?? "#38bdf8"}"></i>${esc(label)}</li>`
      )
      .join("");
  }

  /* -------------------------------------------------------------- api mode */
  async function probe(path, options = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), FALLBACK.probeTimeoutMs);
    try {
      return await fetch(`${state.apiUrl.replace(/\/$/, "")}${path}`, {
        ...options,
        signal: controller.signal,
        cache: "no-store",
      });
    } finally {
      clearTimeout(timer);
    }
  }

  function setStatusPill(mode, health) {
    const pill = $("#api-status");
    pill.classList.remove("pill--live", "pill--demo", "pill--offline");
    const label = $(".pill__label", pill);

    if (mode === "live") {
      pill.classList.add("pill--live");
      label.textContent = `live API v${health?.version ?? "?"}`;
      pill.title = state.apiUrl;
    } else if (mode === "demo") {
      pill.classList.add("pill--demo");
      label.textContent = "demo mode";
      pill.title = `${state.apiUrl} is unreachable — synthetic footprint`;
    } else {
      pill.classList.add("pill--offline");
      label.textContent = "offline";
      pill.title = "API unreachable";
    }
  }

  function applyConfig(config) {
    if (!config) return;
    if (Array.isArray(config.depth_labels) && config.depth_labels.length) {
      state.labels = config.depth_labels.slice();
      renderLegend();
    }
    if (config.terrain_provider) state.provider = config.terrain_provider;
    const radiusInput = $("#radius");
    if (Number.isFinite(Number(config.defaults?.radius_km))) {
      state.defaultRadiusKm = Number(config.defaults.radius_km);
      radiusInput.value = String(state.defaultRadiusKm);
    }
    if (Number.isFinite(Number(config.defaults?.breach_head_m)) && !state.headEdited) {
      $("#breach-head").value = String(Number(config.defaults.breach_head_m));
      $("#breach-output").textContent = `${$("#breach-head").value} m`;
      if (!state.widthEdited) $("#breach-width").value = $("#breach-head").value;
    }

    if (config.limits) {
      if (Number.isFinite(Number(config.limits.max_radius_km))) {
        state.limits.maxRadiusKm = Number(config.limits.max_radius_km);
        radiusInput.max = String(state.limits.maxRadiusKm);
        radiusInput.value = String(Math.min(Number(radiusInput.value), state.limits.maxRadiusKm));
      }
      if (Number.isFinite(Number(config.limits.max_upload_mb))) {
        state.limits.maxUploadMb = Number(config.limits.max_upload_mb);
      }
    }

    paintRange(radiusInput);
    $("#radius-output").textContent = `${radiusInput.value} km`;

    state.limits.uploadAvailable = Boolean(config.upload_available);
    state.uploadReason = config.upload_unavailable_reason || "";
    applyUploadAvailability();
  }

  function applyUploadAvailability() {
    const available = state.limits.uploadAvailable;
    const radio = $('#upload-card input');
    const note = $("#upload-source-note");
    const card = $("#upload-card");

    radio.disabled = !available && state.mode === "live";
    card.classList.toggle("is-disabled", !available && state.mode === "live");
    note.textContent = available
      ? "Use your own DEM raster for a local study."
      : state.mode === "live"
        ? `Unavailable: ${state.uploadReason || "GDAL is not installed on the server"}.`
        : "Available when the full server (with GDAL) is running.";
  }

  function setModeNote() {
    const note = $("#mode-note");
    if (state.mode === "live") {
      note.hidden = true;
      return;
    }
    note.hidden = false;
    note.innerHTML =
      `Demo mode — the API at <code>${esc(state.apiUrl)}</code> is unreachable, so the ` +
      "footprint is synthetic. Start the backend and reload for real terrain: " +
      "<code>cd backend &amp;&amp; uvicorn main:app --port 8010</code>.";
  }

  async function detectMode({ silent = false } = {}) {
    try {
      const response = await probe("/api/health");
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const health = await response.json();
      state.mode = "live";
      setStatusPill("live", health);

      try {
        const configResponse = await probe("/api/config");
        if (configResponse.ok) applyConfig(await configResponse.json());
      } catch {
        /* config is optional */
      }

      if (Number.isFinite(Number(health.max_upload_mb))) {
        state.limits.maxUploadMb = Number(health.max_upload_mb);
      }
      if (health.capabilities) {
        state.limits.uploadAvailable = Boolean(health.capabilities.upload_dem);
        applyUploadAvailability();
      }
    } catch {
      state.mode = "demo";
      setStatusPill("demo");
      applyUploadAvailability();
      if (!silent) toast("API unreachable — running in demo mode.", "warn");
    }
    setModeNote();
    return state.mode;
  }

  async function loadDams() {
    if (state.mode === "live") {
      try {
        const response = await probe("/api/dams");
        if (response.ok) return await response.json();
      } catch {
        /* fall through to the bundled register */
      }
    }
    const response = await fetch("dams.json", { cache: "no-store" });
    if (!response.ok) throw new Error("Could not load the dam register.");
    return await response.json();
  }

  function populateDamSelect(dams) {
    const select = $("#dam");
    select.innerHTML = "";
    dams.forEach((dam) => {
      const option = document.createElement("option");
      option.value = dam.dam_id;
      option.textContent = `${dam.name} — ${dam.state}`;
      select.appendChild(option);
    });
    applySelectedDam();
  }

  function applySelectedDam() {
    const select = $("#dam");
    const dam = state.dams.find((item) => String(item.dam_id) === String(select.value));
    if (!dam) return;
    state.dam = dam;

    const facts = [dam.dam_id, dam.state];
    if (dam.reservoir_level_m) facts.push(`${dam.reservoir_level_m} m MSL`);
    if (dam.storage_mcm) facts.push(`${group(dam.storage_mcm)} MCM`);
    if (dam.dam_height_m) facts.push(`${dam.dam_height_m} m high`);
    $("#dam-meta").textContent = facts.join(" · ");

    // Pre-filling is programmatic, so the values still count as "from the
    // register" — the request only sends them once the user actually edits them.
    if (dam.reservoir_level_m) $("#release-level").value = dam.reservoir_level_m;
    if (dam.storage_mcm) $("#release-volume").value = dam.storage_mcm;
    state.levelEdited = false;
    state.volumeEdited = false;

    $("#register-help").textContent = dam.reservoir_level_m
      ? "From the dam register (indicative published figures) — edit to match dam records. " +
        "The release volume anchors the model: the water-surface gradient is solved so the " +
        "inundation holds exactly this much water."
      : "No register level for this dam, so give a release level or a breach head. " +
        "The release volume anchors the model when supplied.";

    if (mapReady()) {
      map.setView([dam.latitude, dam.longitude], 9);
      updateDamMarker();
    }
  }

  /* ------------------------------------------------------------ demo engine */
  function demoSimulate(inputs) {
    const dam = inputs.dam;
    const rng = mulberry32(hashString(`${dam.dam_id}:${inputs.releaseLevel}:${inputs.radiusKm}`));
    const radiusKm = inputs.radiusKm;
    const bearing = rng() * Math.PI * 2;
    const releaseLevel = inputs.releaseLevel;
    const baseTerrain = releaseLevel - (25 + rng() * 55);
    const gradient = 0.4 + rng() * 1.6;
    const maxDepth = Math.min(90, releaseLevel - baseTerrain);

    const kmPerDegLat = 110.54;
    const kmPerDegLon = 111.32 * Math.cos((dam.latitude * Math.PI) / 180);

    const project = (east, north) => [
      dam.latitude + north / kmPerDegLat,
      dam.longitude + east / kmPerDegLon,
    ];

    const ring = (downstreamKm, widthKm, wobble = 0.14) => {
      const points = 72;
      const phases = [rng() * 6.28, rng() * 6.28];
      const coordinates = [];
      for (let index = 0; index <= points; index += 1) {
        const angle = (index / points) * Math.PI * 2;
        const noise =
          1 +
          wobble *
            (Math.sin(angle * 3 + phases[0]) * 0.6 + Math.sin(angle * 5 + phases[1]) * 0.4);
        const local = { x: Math.cos(angle) * downstreamKm * noise, y: Math.sin(angle) * widthKm * noise };
        const east = local.x * Math.cos(bearing) - local.y * Math.sin(bearing);
        const north = local.x * Math.sin(bearing) + local.y * Math.cos(bearing);
        const [lat, lon] = project(east, north);
        coordinates.push([Number(lon.toFixed(5)), Number(lat.toFixed(5))]);
      }
      return coordinates;
    };

    const features = [];
    for (let band = 0; band < BAND_COLORS.length; band += 1) {
      const shrink = 1 - band * 0.2;
      features.push({
        type: "Feature",
        properties: { severity: BAND_LABELS[band], band, area_km2: 0 },
        geometry: { type: "Polygon", coordinates: [ring(radiusKm * 0.55 * shrink, radiusKm * 0.3 * shrink)] },
      });
    }

    const evaluate = (target) => {
      const east = (target.lon - dam.longitude) * kmPerDegLon;
      const north = (target.lat - dam.latitude) * kmPerDegLat;
      const along = east * Math.cos(bearing) + north * Math.sin(bearing);
      const across = -east * Math.sin(bearing) + north * Math.cos(bearing);
      const distance = haversineKm(dam.latitude, dam.longitude, target.lat, target.lon);
      const inside = distance <= radiusKm * 0.55 && Math.abs(across) <= radiusKm * 0.35;
      const depth = inside ? Math.max(0, maxDepth - gradient * distance) : 0;
      const ground = releaseLevel - depth - distance * 1.2;
      return {
        label: target.label,
        latitude: Number(target.lat.toFixed(5)),
        longitude: Number(target.lon.toFixed(5)),
        inside_grid: true,
        ground_elevation_m: Number(ground.toFixed(1)),
        flood_depth_m: Number(depth.toFixed(2)),
        inundated: depth > 0,
        severity: depth <= 0 ? "dry" : depth < 0.5 ? "minor" : depth < 1 ? "moderate" : depth < 5 ? "severe" : "extreme",
        distance_from_dam_km: Number(distance.toFixed(2)),
      };
    };

    const targets = inputs.targets.map(evaluate);
    const floodedAreaKm2 = Math.PI * (radiusKm * 0.55) * (radiusKm * 0.3);

    // Synthetic profile: a bed that climbs away from the dam and a surface that
    // falls at the solved gradient, so the demo still shows where the flood ends.
    const reachKm = Math.max(2, radiusKm);
    const edgeKm = Math.max(1.0, radiusKm * 0.55);
    const bedSlope = maxDepth / edgeKm - gradient;
    const profileSamples = 161;
    const profileDistance = [];
    const profileTerrain = [];
    const profileSurface = [];
    const profileDepth = [];
    const profileFlooded = [];
    for (let index = 0; index < profileSamples; index += 1) {
      const distance = (reachKm * index) / (profileSamples - 1);
      const surface = releaseLevel - gradient * distance;
      const ground = releaseLevel - maxDepth + bedSlope * distance;
      const wet = ground < surface;
      profileDistance.push(Number(distance.toFixed(3)));
      profileSurface.push(Number(surface.toFixed(2)));
      profileTerrain.push(Number(ground.toFixed(1)));
      profileDepth.push(wet ? Number((surface - ground).toFixed(2)) : 0);
      profileFlooded.push(wet);
    }

    // Synthetic cross-section: a valley that closes off a couple of kilometres
    // either side of each point.
    const sections = targets.map((target) => {
      const halfWidthKm = 3;
      const count = 161;
      const bed = Number(target.ground_elevation_m);
      const level = bed + Number(target.flood_depth_m || 0);
      const halfWetKm = Math.max(0.25, Math.sqrt(Number(target.flood_depth_m || 0) / 0.6));
      const curvature = Number(target.flood_depth_m || 0) / (halfWetKm * halfWetKm);
      const offsets = [];
      const terrainValues = [];
      const surfaceValues = [];
      const depthValues = [];
      const floodedFlags = [];
      for (let index = 0; index < count; index += 1) {
        const offset = -halfWidthKm + (2 * halfWidthKm * index) / (count - 1);
        const ground = bed + curvature * offset * offset;
        const wet = Number(target.flood_depth_m || 0) > 0 && ground < level;
        offsets.push(Number(offset.toFixed(3)));
        terrainValues.push(Number(ground.toFixed(1)));
        surfaceValues.push(Number(level.toFixed(2)));
        depthValues.push(wet ? Number((level - ground).toFixed(2)) : 0);
        floodedFlags.push(wet);
      }
      return {
        label: target.label,
        latitude: target.latitude,
        longitude: target.longitude,
        x_label: "Distance across the valley (km)",
        y_label: "Elevation (m MSL)",
        half_width_km: halfWidthKm,
        span_km: halfWidthKm * 2,
        flood_width_km: Number((halfWetKm * 2).toFixed(2)),
        offset_km: offsets,
        terrain_m: terrainValues,
        water_surface_m: surfaceValues,
        depth_m: depthValues,
        flooded: floodedFlags,
      };
    });

    const bandShare = [0.04, 0.04, 0.08, 0.18, 0.66];

    return {
      synthetic: true,
      dam: {
        dam_id: dam.dam_id,
        name: dam.name,
        state: dam.state,
        latitude: dam.latitude,
        longitude: dam.longitude,
      },
      inputs: {
        terrain_source: "auto",
        release_level_m: releaseLevel,
        release_level_source: "register",
        release_volume_mcm: inputs.releaseVolume,
        release_volume_source: "register",
        pool_level_m: releaseLevel,
        volume_matched: true,
        impounded_volume_mcm: inputs.releaseVolume,
        breach_head_m: inputs.breachHead,
        breach_width_m: inputs.breachWidth,
        attenuation_m_per_km: gradient,
        attenuation_source: "solved",
        flow_direction_east_north: [Math.cos(bearing), Math.sin(bearing)],
        radius_km: radiusKm,
        dam_snapped_to_nearest_cell: false,
        dam_snap_distance_km: 0,
      },
      summary: {
        estimated_flooded_area_km2: Number(floodedAreaKm2.toFixed(2)),
        maximum_depth_m: Number(maxDepth.toFixed(2)),
        mean_depth_m: Number((maxDepth * 0.38).toFixed(2)),
        source_elevation_m: Number(baseTerrain.toFixed(1)),
        water_surface_m: releaseLevel,
        pool_level_m: releaseLevel,
        release_level_m: releaseLevel,
        impounded_volume_mcm: inputs.releaseVolume,
        volume_matched: true,
        flood_volume_mcm: inputs.releaseVolume,
        peak_discharge_m3s: Number(
          ((8 / 27) * inputs.breachWidth * Math.sqrt(9.80665) * inputs.breachHead ** 1.5).toFixed(1)
        ),
        flooded_cells: Math.round(floodedAreaKm2 * 235),
        depth_stats: {
          mean_m: Number((maxDepth * 0.38).toFixed(3)),
          median_m: Number((maxDepth * 0.3).toFixed(3)),
          p95_m: Number((maxDepth * 0.85).toFixed(3)),
          max_m: Number(maxDepth.toFixed(3)),
        },
        grid_rows: 620,
        grid_columns: 620,
        cell_size_x_m: 65.2,
        cell_size_y_m: 65.2,
        cell_area_m2: 4251,
        depth_band_area: {
          labels: BAND_LABELS,
          area_km2: bandShare.map((share) =>
            Number((floodedAreaKm2 * share).toFixed(4))),
        },
      },
      terrain: {
        source: "auto",
        provider: "Demo generator (no API reachable)",
        provider_url: null,
        zoom: 11,
        resolution_m: 65.2,
        native_resolution_m: 65.2,
        downsample_factor: 1,
        mosaic_rows: 620,
        mosaic_columns: 620,
        tiles_requested: 12,
        tiles_failed: 0,
        bounds_wgs84: null,
        elevation_min_m: Number(baseTerrain.toFixed(1)),
        elevation_max_m: Number((baseTerrain + 480).toFixed(1)),
        nodata_fraction: 0,
        radius_km: radiusKm,
        vertical_check: null,
      },
      targets,
      severity_legend: BAND_LABELS,
      plots: {
        profile: {
          x_label: "Distance downstream from the dam (km)",
          y_label: "Elevation (m MSL)",
          reach_km: Number(reachKm.toFixed(2)),
          gradient_m_per_km: Number(gradient.toFixed(3)),
          gradient_source: "solved",
          release_level_m: releaseLevel,
          flow_direction_east_north: [Math.cos(bearing), Math.sin(bearing)],
          distance_km: profileDistance,
          latitude: null,
          longitude: null,
          terrain_m: profileTerrain,
          water_surface_m: profileSurface,
          depth_m: profileDepth,
          flooded: profileFlooded,
        },
        sections,
      },
      flood_geojson: { type: "FeatureCollection", features, bbox: null },
      exposure: null,
      grid3d: demoGrid3d(),
      notes: [
        { level: "warning", text: "DEMO MODE: this footprint is synthetic, not derived from terrain. Start the backend for real results." },
      ],
      processing_ms: Math.round(140 + rng() * 400),
      warning:
        "Screening approximation only; not an engineering-grade hydraulic prediction. " +
        "Do not use for evacuation, dam-safety certification or regulatory decisions.",
    };
  }

  // Demo fallback: build a small synthetic grid3d too, so the 3-D button
  // explains itself offline instead of dead-ending. Labelled synthetic.
  function demoGrid3d() {
    const rows = 60;
    const cols = 60;
    const cell = 65.2;
    const elevation = [];
    const depth = [];
    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        const valley = Math.abs(col - cols / 2) / (cols / 2);
        const elev = 320 - row * 2.2 + valley * valley * 130;
        elevation.push(Math.round(elev * 10));
        const wet = row > 8 && valley < 0.45 ? Math.max(0, (0.45 - valley) * 22) : 0;
        depth.push(Math.round(wet * 10));
      }
    }
    return {
      rows,
      cols,
      step_cells: 10,
      cell_size_x_m: cell * 10,
      cell_size_y_m: cell * 10,
      dam_row: 6,
      dam_col: 30,
      dam_height_m: null,
      release_level_m: 500,
      pool_level_m: 500,
      elevation_decim: elevation,
      depth_decim: depth,
      scale: 10.0,
      nodata: -999999,
    };
  }
  /* -------------------------------------------------------------- live run */
  function collectInputs() {
    const head = Number($("#breach-head").value);
    const width = Number($("#breach-width").value);
    const mode = $("#attenuation-mode").value;
    return {
      dam: state.dam,
      releaseLevel: Number($("#release-level").value),
      releaseVolume: Number($("#release-volume").value),
      breachHead: head,
      breachWidth: Number.isFinite(width) && width > 0 ? width : head,
      radiusKm: Number($("#radius").value),
      attenuation: mode === "manual" ? Number($("#attenuation").value) : null,
      includeExposure: $("#exposure-toggle").checked,
      verify: $("#verify-toggle").checked,
      targets: [
        { label: "Area 1", lat: Number($("#area1-lat").value), lon: Number($("#area1-lon").value) },
        { label: "Area 2", lat: Number($("#area2-lat").value), lon: Number($("#area2-lon").value) },
      ],
    };
  }

  function validate(inputs) {
    if (!state.dam) return "Select a dam first.";
    if (!Number.isFinite(inputs.releaseLevel) && !Number.isFinite(inputs.breachHead)) {
      return "Provide a release level or a breach head.";
    }
    for (const target of inputs.targets) {
      if (!Number.isFinite(target.lat) || !Number.isFinite(target.lon)) {
        return `${target.label} needs valid coordinates.`;
      }
      if (Math.abs(target.lat) > 90 || Math.abs(target.lon) > 180) {
        return `${target.label} is outside valid latitude/longitude bounds.`;
      }
    }
    if (state.terrainSource === "upload" && state.mode === "live") {
      if (!state.limits.uploadAvailable) return "This server cannot read GeoTIFF uploads (no GDAL).";
      if (!state.demFile) return "Attach a GeoTIFF DEM before running an uploaded-terrain simulation.";
      if (state.demFile.size > state.limits.maxUploadMb * 1024 * 1024) {
        return `The DEM is larger than the ${state.limits.maxUploadMb} MB server limit.`;
      }
    }
    return null;
  }

  async function runAuto(inputs) {
    const body = {
      dam_id: inputs.dam.dam_id,
      radius_km: inputs.radiusKm,
      area1_lat: inputs.targets[0].lat,
      area1_lon: inputs.targets[0].lon,
      area2_lat: inputs.targets[1].lat,
      area2_lon: inputs.targets[1].lon,
      verify_vertical: inputs.verify,
      include_exposure: inputs.includeExposure,
      breach_head_m: inputs.breachHead,
      breach_width_m: inputs.breachWidth,
    };
    // Only send these when the user overrode the register value, so the server's
    // provenance stays truthful ("register" vs "request").
    if (state.levelEdited && Number.isFinite(inputs.releaseLevel)) {
      body.release_level_m = inputs.releaseLevel;
    }
    if (state.volumeEdited && Number.isFinite(inputs.releaseVolume) && inputs.releaseVolume > 0) {
      body.release_volume_mcm = inputs.releaseVolume;
    }
    if (inputs.attenuation !== null) body.attenuation_m_per_km = inputs.attenuation;

    const response = await fetch(`${state.apiUrl.replace(/\/$/, "")}/api/simulate/auto`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || `Simulation failed (HTTP ${response.status}).`);
    return data;
  }

  async function runUpload(inputs) {
    const form = new FormData();
    form.append("dam_id", inputs.dam.dam_id);
    form.append("dem_file", state.demFile, state.demFile.name);
    form.append("area1_lat", String(inputs.targets[0].lat));
    form.append("area1_lon", String(inputs.targets[0].lon));
    form.append("area2_lat", String(inputs.targets[1].lat));
    form.append("area2_lon", String(inputs.targets[1].lon));
    form.append("breach_head_m", String(inputs.breachHead));
    form.append("breach_width_m", String(inputs.breachWidth));
    if (state.levelEdited && Number.isFinite(inputs.releaseLevel)) {
      form.append("release_level_m", String(inputs.releaseLevel));
    }
    if (state.volumeEdited && Number.isFinite(inputs.releaseVolume)) {
      form.append("release_volume_mcm", String(inputs.releaseVolume));
    }
    if (inputs.attenuation !== null) form.append("attenuation_m_per_km", String(inputs.attenuation));

    const response = await fetch(`${state.apiUrl.replace(/\/$/, "")}/api/simulate`, {
      method: "POST",
      body: form,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || `Simulation failed (HTTP ${response.status}).`);
    return data;
  }

  /* --------------------------------------------------------------- renders */
  function statCard(label, value, unit, decimals = 2, extra = "") {
    return `
      <div class="stat">
        <span class="stat__label">${esc(label)}</span>
        <span class="stat__value" data-count="${Number(value) || 0}" data-decimals="${decimals}">0</span>
        <span class="stat__unit">${esc(unit)}</span>
        ${extra ? `<span class="stat__extra">${extra}</span>` : ""}
      </div>`;
  }

  function renderTargets(targets) {
    return targets
      .map((target) => {
        const badge = SEVERITY_BADGE[target.severity] || SEVERITY_BADGE.dry;
        const depth = Number(target.flood_depth_m) || 0;
        const percent = clamp((depth / 10) * 100, 0, 100);
        return `
          <tr>
            <td>${esc(target.label)}</td>
            <td>${fmt(target.distance_from_dam_km, 1)} km</td>
            <td>${fmt(target.ground_elevation_m, 1)} m</td>
            <td>${fmt(depth, 2)} m</td>
            <td><span class="badge ${badge}">${target.inundated ? esc(target.severity) : "dry"}</span></td>
            <td class="col-bar"><div class="depth-bar"><i style="width:${percent}%"></i></div></td>
          </tr>`;
      })
      .join("");
  }

  function renderProvenance(payload) {
    const terrain = payload.terrain || {};
    const rows = [
      ["Source", terrain.provider || "—"],
      ["Resolution", terrain.resolution_m ? `${terrain.resolution_m} m / px` : "—"],
      ["Analysis radius", terrain.radius_km ? `${terrain.radius_km} km` : "—"],
      [
        "Mosaic",
        terrain.mosaic_rows ? `${terrain.mosaic_columns} × ${terrain.mosaic_rows} px` : "—",
      ],
      [
        "Tiles",
        Number.isFinite(Number(terrain.tiles_requested))
          ? `${terrain.tiles_requested} fetched${terrain.tiles_failed ? `, ${terrain.tiles_failed} failed` : ""}`
          : "—",
      ],
      [
        "Elevation",
        terrain.elevation_min_m != null && terrain.elevation_max_m != null
          ? `${terrain.elevation_min_m} – ${terrain.elevation_max_m} m`
          : "—",
      ],
      ["Model grid", `${payload.summary.grid_columns} × ${payload.summary.grid_rows} px`],
      ["Cell size", `${fmt(payload.summary.cell_size_x_m, 1)} m`],
    ];

    const check = terrain.vertical_check;
    const checkRow = check
      ? `<p class="provenance__check">
           <strong>Vertical cross-check:</strong> mean |Δ| ${esc(check.mean_abs_delta_m)} m,
           max ${esc(check.max_abs_delta_m)} m against ${esc(check.reference)}
           (${check.samples} point${check.samples === 1 ? "" : "s"}).
         </p>`
      : "";

    return `
      <table class="table table--kv">
        ${rows
          .map(([label, value]) => `<tr><th>${esc(label)}</th><td>${esc(value)}</td></tr>`)
          .join("")}
      </table>
      ${checkRow}`;
  }

  function renderExposure(exposure) {
    if (!exposure) return "";
    if (exposure.available === false) {
      const reason = String(exposure.reason || "OpenStreetMap lookup failed").replace(/\.+$/, "");
      return `
        <p class="field__help">
          Asset exposure unavailable: ${esc(reason)}. The flood result above is unaffected —
          only the asset breakdown could not be fetched.
        </p>`;
    }

    const rows = exposure.groups
      .filter((groupEntry) => groupEntry.total > 0)
      .map(
        (groupEntry) => `
        <tr>
          <td>${esc(groupEntry.label)}</td>
          <td class="col-bar">
            <span class="exposure-bar" title="${group(groupEntry.inundated)} of ${group(groupEntry.total)} inside the flood" aria-hidden="true">
              <i style="width:${((groupEntry.inundated / Math.max(groupEntry.total, 1)) * 100).toFixed(0)}%"></i>
            </span>
          </td>
          <td>${group(groupEntry.total)}</td>
          <td class="${groupEntry.inundated ? "cell-alert" : ""}">${group(groupEntry.inundated)}</td>
          <td>${groupEntry.inundated ? `${fmt(groupEntry.max_depth_m, 1)} m` : "—"}</td>
        </tr>`
      )
      .join("");

    const examples = exposure.groups
      .flatMap((groupEntry) => groupEntry.examples || [])
      .slice(0, 6)
      .map((name) => `<span class="chip chip--alert">${esc(name)}</span>`)
      .join("");

    return `
      <div class="exposure">
        <p class="exposure__headline">
          <strong>${group(exposure.assets_inundated)}</strong> of
          ${group(exposure.assets_total)} mapped assets fall inside the flood.
        </p>
        <table class="table">
          <thead><tr><th>Category</th><th class="sr-only">Share in flood</th><th>Mapped</th><th>In flood</th><th>Max depth</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
        ${examples ? `<div class="results__foot">${examples}</div>` : ""}
        <p class="field__help">${esc(exposure.source)} · ${esc(exposure.licence || "")}</p>
      </div>`;
  }

  function renderNotes(notes = []) {
    if (!notes.length) return "";
    return `
      <ul class="notes">
        ${notes
          .map(
            (note) => `
          <li class="notes__item notes__item--${esc(note.level)}">
            <span class="notes__mark" aria-hidden="true"></span>
            <span>${esc(note.text)}</span>
          </li>`
          )
          .join("")}
      </ul>`;
  }

  /* --------------------------------------------------------------- charts */
  // Hand-written SVG rather than a chart library: it prints, themes off the same
  // CSS variables as the rest of the page, and adds no dependency to ship.
  const CHART_W = 760;
  const CHART_H = 300;
  const CHART_PAD = { top: 20, right: 18, bottom: 50, left: 62 };

  function flagRuns(flags) {
    const out = [];
    let start = -1;
    flags.forEach((flag, index) => {
      if (flag && start < 0) start = index;
      else if (!flag && start >= 0) {
        out.push([start, index - 1]);
        start = -1;
      }
    });
    if (start >= 0) out.push([start, flags.length - 1]);
    return out;
  }

  function tickSpan(min, max, count = 4) {
    if (!Number.isFinite(min) || !Number.isFinite(max)) return null;
    if (max - min < 1e-6) {
      min -= 1;
      max += 1;
    }
    const pad = (max - min) * 0.07;
    min -= pad;
    max += pad;
    const step = (max - min) / count;
    return {
      min,
      max,
      decimals: step >= 10 ? 0 : step >= 1 ? 1 : 2,
      ticks: Array.from({ length: count + 1 }, (_, index) =>
        min + ((max - min) * index) / count),
    };
  }

  function pathFrom(points) {
    return points
      .map(([x, y], index) => `${index ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`)
      .join(" ");
  }

  /**
   * Terrain profile with the modelled water surface over it.
   *
   * spec: { x, terrain, surface, flooded, xLabel, alt, unit, reference,
   *         markers: [{ at, text }] }
   */
  function renderElevationChart(spec) {
    // null/undefined samples must stay NaN, not become 0: a missing elevation
    // silently plotted at sea level would be a lie the chart cannot admit to.
    const toNum = (value) => (value == null || value === "" ? NaN : Number(value));
    const series = [];
    for (let index = 0; index < spec.x.length; index += 1) {
      const x = toNum(spec.x[index]);
      if (!Number.isFinite(x)) continue;
      series.push({
        x,
        terrain: toNum(spec.terrain[index]),
        surface: toNum(spec.surface[index]),
        flooded: Boolean(spec.flooded && spec.flooded[index]),
      });
    }
    const xValues = series.map((point) => point.x).filter(Number.isFinite);
    if (xValues.length < 2) return "";

    const heights = [];
    series.forEach((point) => {
      if (Number.isFinite(point.terrain)) heights.push(point.terrain);
      if (Number.isFinite(point.surface)) heights.push(point.surface);
    });
    if (spec.reference && Number.isFinite(spec.reference.value)) {
      heights.push(spec.reference.value);
    }
    if (!heights.length) return "";

    const ySpan = tickSpan(Math.min(...heights), Math.max(...heights));
    if (!ySpan) return "";
    const xMin = Math.min(...xValues);
    const xMax = Math.max(...xValues);

    const innerW = CHART_W - CHART_PAD.left - CHART_PAD.right;
    const innerH = CHART_H - CHART_PAD.top - CHART_PAD.bottom;
    const sx = (value) =>
      CHART_PAD.left + ((value - xMin) / (xMax - xMin || 1)) * innerW;
    const sy = (value) =>
      CHART_H - CHART_PAD.bottom - ((value - ySpan.min) / (ySpan.max - ySpan.min || 1)) * innerH;

    const groundRuns = flagRuns(series.map((point) => Number.isFinite(point.terrain)));
    const wetRuns = flagRuns(
      series.map((point) => point.flooded && Number.isFinite(point.surface))
    );

    const parts = [
      `<svg class="chart__svg" viewBox="0 0 ${CHART_W} ${CHART_H}" role="img" ` +
        `aria-label="${esc(spec.alt || "Elevation profile")}" preserveAspectRatio="xMidYMid meet">`,
    ];

    ySpan.ticks.forEach((tick) => {
      const py = sy(tick);
      parts.push(
        `<line class="chart__grid" x1="${CHART_PAD.left}" y1="${py.toFixed(1)}" ` +
          `x2="${CHART_W - CHART_PAD.right}" y2="${py.toFixed(1)}"/>`,
        `<text class="chart__tick" x="${CHART_PAD.left - 9}" y="${(py + 4).toFixed(1)}" ` +
          `text-anchor="end">${esc(fmt(tick, ySpan.decimals))}</text>`
      );
    });

    const xTickCount = 5;
    for (let index = 0; index <= xTickCount; index += 1) {
      const value = xMin + ((xMax - xMin) * index) / xTickCount;
      const px = sx(value);
      parts.push(
        `<text class="chart__tick" x="${px.toFixed(1)}" y="${CHART_H - CHART_PAD.bottom + 20}" ` +
          `text-anchor="middle">${esc(fmt(value, value >= 100 ? 0 : 1))}</text>`
      );
    }

    // Ground: fill down to the axis so the profile reads as solid rock.
    groundRuns.forEach(([start, end]) => {
      const line = [];
      for (let index = start; index <= end; index += 1) {
        const point = series[index];
        if (!Number.isFinite(point.terrain)) continue;
        line.push([sx(point.x), sy(point.terrain)]);
      }
      if (line.length < 2) return;
      const floorY = CHART_H - CHART_PAD.bottom;
      const area = `${pathFrom(line)} L${line[line.length - 1][0].toFixed(1)},${floorY} ` +
        `L${line[0][0].toFixed(1)},${floorY} Z`;
      parts.push(`<path class="chart__ground" d="${area}"/>`);
      parts.push(`<path class="chart__line chart__line--ground" d="${pathFrom(line)}"/>`);
    });

    // Water: only over the inundated reach, so it ends where the flood ends.
    wetRuns.forEach(([start, end]) => {
      const top = [];
      const bottom = [];
      for (let index = start; index <= end; index += 1) {
        const point = series[index];
        if (!Number.isFinite(point.surface) || !Number.isFinite(point.terrain)) continue;
        top.push([sx(point.x), sy(point.surface)]);
        bottom.push([sx(point.x), sy(point.terrain)]);
      }
      if (top.length < 2) return;
      const area = `${pathFrom(top)} ${bottom
        .slice()
        .reverse()
        .map(([x, y]) => `L${x.toFixed(1)},${y.toFixed(1)}`)
        .join(" ")} Z`;
      let deepest = 0;
      for (let index = start; index <= end; index += 1) {
        const point = series[index];
        if (Number.isFinite(point.surface) && Number.isFinite(point.terrain)) {
          deepest = Math.max(deepest, point.surface - point.terrain);
        }
      }
      parts.push(
        `<path class="chart__water" d="${area}">` +
          `<title>Water surface over the reach, up to ${esc(fmt(deepest, 1))} m above the ground</title>` +
          `</path>`,
        `<path class="chart__line chart__line--water" d="${pathFrom(top)}"/>`
      );
    });

    if (spec.reference && Number.isFinite(spec.reference.value)) {
      const py = sy(spec.reference.value);
      parts.push(
        `<line class="chart__ref" x1="${CHART_PAD.left}" y1="${py.toFixed(1)}" ` +
          `x2="${CHART_W - CHART_PAD.right}" y2="${py.toFixed(1)}"/>`,
        `<text class="chart__reflabel" x="${CHART_W - CHART_PAD.right}" ` +
          `y="${(py - 7).toFixed(1)}" text-anchor="end">${esc(spec.reference.label)}</text>`
      );
    }

    const innerRight = CHART_W - CHART_PAD.right;
    const placedMarkers = [];
    (spec.markers || []).forEach((marker) => {
      if (!Number.isFinite(marker.at)) return;
      const px = sx(marker.at);
      // Near the right edge a left-anchored label would run out of the plot and
      // be clipped, so it flips to right-aligned and sits inside the frame.
      const flip = px > innerRight - CHART_W * 0.28;
      // Two markers close together would print their labels on top of each
      // other, so the second one drops to a lower row.
      const row = placedMarkers.some((other) => Math.abs(other - px) < CHART_W * 0.24) ? 1 : 0;
      placedMarkers.push(px);
      parts.push(
        `<line class="chart__marker" x1="${px.toFixed(1)}" y1="${CHART_PAD.top}" ` +
          `x2="${px.toFixed(1)}" y2="${CHART_H - CHART_PAD.bottom}"/>`,
        `<text class="chart__markerlabel" x="${(px + (flip ? -6 : 6)).toFixed(1)}" ` +
          `y="${CHART_PAD.top + 14 + row * 17}"${flip ? ' text-anchor="end"' : ""}>` +
          `${esc(marker.text)}</text>`
      );
    });

    parts.push(
      `<text class="chart__axislabel" x="${CHART_W / 2}" y="${CHART_H - 8}" ` +
        `text-anchor="middle">${esc(spec.xLabel || "")}</text>`,
      `<text class="chart__axislabel" transform="rotate(-90 14 ${(CHART_H - CHART_PAD.bottom) / 2 + CHART_PAD.top})" ` +
        `x="14" y="${(CHART_H - CHART_PAD.bottom) / 2 + CHART_PAD.top}" ` +
        `text-anchor="middle">${esc(spec.yLabel || "Elevation (m MSL)")}</text>`,
      `<line class="chart__axis" x1="${CHART_PAD.left}" y1="${CHART_H - CHART_PAD.bottom}" ` +
        `x2="${CHART_W - CHART_PAD.right}" y2="${CHART_H - CHART_PAD.bottom}"/>`,
      `</svg>`
    );

    return parts.join("");
  }

  function renderProfileChart(profile) {
    if (!profile || !Array.isArray(profile.distance_km) || profile.distance_km.length < 2) {
      return "";
    }
    const wet = profile.distance_km.filter((_, index) => profile.flooded[index]);
    const edge = wet.length ? wet[wet.length - 1] : null;
    const markers = [];
    if (edge != null && edge < profile.distance_km[profile.distance_km.length - 1]) {
      markers.push({ at: edge, text: `flood ends ${fmt(edge, 1)} km` });
    }

    const gradient =
      profile.gradient_source === "solved"
        ? `solved gradient ${fmt(profile.gradient_m_per_km, 2)} m/km`
        : profile.gradient_source === "user"
          ? `gradient ${fmt(profile.gradient_m_per_km, 2)} m/km (set)`
          : "flat pool";

    return `
      <div class="chart" role="group" aria-label="Longitudinal profile">
        <div class="chart__head">
          <h4 class="section-label">Longitudinal profile</h4>
          <span class="chip">${esc(gradient)}</span>
        </div>
        ${renderElevationChart({
          x: profile.distance_km,
          terrain: profile.terrain_m,
          surface: profile.water_surface_m,
          flooded: profile.flooded,
          xLabel: profile.x_label,
          yLabel: profile.y_label,
          alt: "Terrain and modelled water surface from the dam to the edge of the analysis window",
          reference: {
            value: profile.release_level_m,
            label: `release level ${fmt(profile.release_level_m, 1)} m`,
          },
          markers,
        })}
        <p class="chart__note">
          Straight line along the solved flow direction from the dam, ${fmt(profile.reach_km, 1)} km to
          the edge of the window — the valley may meander around it. The surface falls linearly by
          ${fmt(profile.gradient_m_per_km, 3)} m/km and is drawn only over the inundated reach.
        </p>
      </div>`;
  }

  function renderSectionChart(section) {
    if (!section || !Array.isArray(section.offset_km) || section.offset_km.length < 2) {
      return "";
    }
    const wet = section.offset_km.filter((_, index) => section.flooded[index]);
    const markers = wet.length
      ? [
          { at: 0, text: section.label },
          { at: wet[wet.length - 1], text: `edge ${fmt(section.flood_width_km, 2)} km wide` },
        ]
      : [{ at: 0, text: section.label }];

    return `
      <div class="chart" role="group" aria-label="Cross-section ${esc(section.label)}">
        <div class="chart__head">
          <h4 class="section-label">Cross-section · ${esc(section.label)}</h4>
          <span class="chip">${fmt(section.latitude, 4)}, ${fmt(section.longitude, 4)}</span>
        </div>
        ${renderElevationChart({
          x: section.offset_km,
          terrain: section.terrain_m,
          surface: section.water_surface_m,
          flooded: section.flooded,
          xLabel: section.x_label,
          yLabel: section.y_label,
          alt: `Cross-section through ${section.label}`,
          markers,
        })}
        <p class="chart__note">
          Cut across the flow at the point, ${fmt(section.span_km, 1)} km wide. Flooded width
          ${fmt(section.flood_width_km, 2)} km.
        </p>
      </div>`;
  }

  function renderBandBars(bandArea) {
    if (!bandArea || !Array.isArray(bandArea.area_km2) || !bandArea.area_km2.length) return "";
    const max = Math.max(...bandArea.area_km2, 1e-9);
    const total = bandArea.area_km2.reduce((sum, value) => sum + value, 0) || 1;

    const rows = bandArea.area_km2
      .map((value, index) => {
        const label = bandArea.labels[index] || `band ${index}`;
        const width = (value / max) * 100;
        const share = (value / total) * 100;
        return `
          <li class="bandbar">
            <span class="bandbar__label">${esc(label)}</span>
            <span class="bandbar__track">
              <i class="bandbar__fill" style="width:${width.toFixed(1)}%;--swatch:${BAND_COLORS[index] || "#38bdf8"}" aria-hidden="true"></i>
            </span>
            <span class="bandbar__value">${fmt(value, 2)} km²</span>
            <span class="bandbar__share">${fmt(share, 1)}%</span>
          </li>`;
      })
      .join("");

    return `
      <div class="chart" role="group" aria-label="Depth band areas">
        <div class="chart__head">
          <h4 class="section-label">Where the water goes</h4>
          <span class="chip">${fmt(total, 1)} km² total</span>
        </div>
        <ul class="bandbars">${rows}</ul>
        <p class="chart__note">
          Flooded area split by depth band — the same cells that make up the headline total,
          summed from the model grid rather than the drawn polygons.
        </p>
      </div>`;
  }

  function renderResults(payload) {
    const summary = payload.summary;
    const inputs = payload.inputs || {};
    const stats = summary.depth_stats || {};
    const body = $("#results-body");

    const gradientLabel =
      inputs.attenuation_source === "solved"
        ? `${fmt(inputs.attenuation_m_per_km, 2)} m/km <em>(solved)</em>`
        : inputs.attenuation_source === "user"
          ? `${fmt(inputs.attenuation_m_per_km, 2)} m/km <em>(set)</em>`
          : "flat pool";

    const budgetNote =
      inputs.release_volume_mcm != null
        ? `of ${group(inputs.release_volume_mcm)} MCM budget${inputs.volume_matched ? "" : " — window limited"}`
        : "no volume budget given";

    const chips = [
      `${fmt(payload.processing_ms, 0)} ms`,
      `${summary.flooded_cells ? group(summary.flooded_cells) : 0} cells`,
      `flow ${(inputs.flow_direction_east_north || []).map((v) => fmt(v, 2)).join(", ")}`,
      payload.synthetic ? "synthetic" : "measured terrain",
      inputs.dam_snapped_to_nearest_cell ? `dam snapped ${fmt(inputs.dam_snap_distance_km, 1)} km` : null,
    ].filter(Boolean);

    body.innerHTML = `
      <div class="results__title">
        <h3>${esc(payload.dam.name)}</h3>
        <span>${esc(payload.dam.state)}</span>
      </div>

      <div class="stat-grid">
        ${statCard("Flooded area", summary.estimated_flooded_area_km2, "km²", 2)}
        ${statCard("Max depth", summary.maximum_depth_m, "m", 1)}
        ${statCard("Peak discharge", summary.peak_discharge_m3s, "m³/s", 0)}
        ${statCard("Impounded volume", summary.impounded_volume_mcm, "MCM", 1, esc(budgetNote))}
        <div class="stat stat--wide">
          <span class="stat__label">Modelled pool</span>
          <p class="stat__line">
            Release level <strong>${fmt(inputs.release_level_m ?? summary.release_level_m, 1)} m</strong>
            <span class="unit">(${esc(inputs.release_level_source || "—")})</span> ·
            surface gradient <strong>${gradientLabel}</strong> ·
            breach <strong>${fmt(inputs.breach_head_m, 0)} m</strong> deep,
            <strong>${fmt(inputs.breach_width_m, 0)} m</strong> wide
          </p>
        </div>
        <div class="stat stat--wide">
          <span class="stat__label">Depth distribution (within flood)</span>
          <table class="table">
            <tr><th>mean</th><th>median</th><th>p95</th><th>max</th></tr>
            <tr>
              <td>${fmt(stats.mean_m)} m</td>
              <td>${fmt(stats.median_m)} m</td>
              <td>${fmt(stats.p95_m)} m</td>
              <td>${fmt(stats.max_m)} m</td>
            </tr>
          </table>
        </div>
      </div>

      <h4 class="section-label">Areas of interest</h4>
      <table class="table">
        <thead>
          <tr><th>Point</th><th>Distance</th><th>Ground</th><th>Depth</th><th>Verdict</th><th></th></tr>
        </thead>
        <tbody>${renderTargets(payload.targets || [])}</tbody>
      </table>

      ${renderBandBars(summary.depth_band_area)}

      ${renderProfileChart(payload.plots?.profile)}

      ${(payload.plots?.sections || [])
        .map((section) => renderSectionChart(section))
        .join("")}

      <h4 class="section-label">Terrain provenance</h4>
      ${renderProvenance(payload)}

      ${
        payload.exposure
          ? `<h4 class="section-label">Asset exposure</h4>${renderExposure(payload.exposure)}`
          : ""
      }

      <h4 class="section-label">Model caveats</h4>
      ${renderNotes(payload.notes)}
      <p class="banner banner--warning">${esc(payload.warning)}</p>

      <div class="results__foot">
        ${chips.map((chip) => `<span class="chip">${esc(chip)}</span>`).join("")}
      </div>
    `;

    $("#results-empty").hidden = true;
    body.hidden = false;
    $$("[data-count]", body).forEach((element) =>
      animateNumber(element, Number(element.dataset.count), Number(element.dataset.decimals))
    );
  }

  function renderSummaryChips(payload) {
    const summary = payload.summary;
    const chips = $("#summary-chips");
    chips.hidden = false;
    chips.innerHTML = `
      <span class="chip"><strong>${fmt(summary.estimated_flooded_area_km2, 1)}</strong> km² flooded</span>
      <span class="chip"><strong>${fmt(summary.maximum_depth_m, 1)}</strong> m max depth</span>
      <span class="chip"><strong>${group(summary.peak_discharge_m3s)}</strong> m³/s peak Q</span>
    `;
  }

  /* ------------------------------------------------------------ print report */
  function reportMetaPairs(payload) {
    const summary = payload.summary || {};
    const inputs = payload.inputs || {};
    const terrain = payload.terrain || {};
    const stats = summary.depth_stats || {};

    return [
      ["Terrain", terrain.provider || "—"],
      ["Resolution", terrain.resolution_m ? `${fmt(terrain.resolution_m, 1)} m/px` : "—"],
      ["Analysis radius", terrain.radius_km ? `${fmt(terrain.radius_km, 0)} km` : "—"],
      [
        "Release level",
        inputs.release_level_m != null
          ? `${fmt(inputs.release_level_m, 1)} m MSL (${inputs.release_level_source || "—"})`
          : "—",
      ],
      [
        "Release volume",
        inputs.release_volume_mcm != null
          ? `${group(inputs.release_volume_mcm)} MCM (${inputs.release_volume_source || "—"})`
            + (inputs.volume_matched ? "" : " · window limited")
          : "none — flat-pool upper bound",
      ],
      [
        "Breach",
        `${fmt(inputs.breach_head_m, 0)} m deep × ${fmt(inputs.breach_width_m, 0)} m wide`,
      ],
      ["Flooded area", `${fmt(summary.estimated_flooded_area_km2, 2)} km²`],
      ["Maximum depth", `${fmt(summary.maximum_depth_m, 1)} m`],
      ["Mean / p95 depth", `${fmt(stats.mean_m, 1)} m / ${fmt(stats.p95_m, 1)} m`],
      ["Peak discharge", `${group(summary.peak_discharge_m3s)} m³/s`],
      ["Impounded volume", `${fmt(summary.impounded_volume_mcm, 1)} MCM`],
      ["Model grid", `${summary.grid_rows || 0} × ${summary.grid_columns || 0} cells`],
    ];
  }

  function renderPrintHead(payload) {
    const dam = payload.dam || {};
    const head = $("#print-head");
    const scenario = $("#print-scenario");
    if (!head || !scenario) return;

    const generated = new Date().toLocaleString(undefined, {
      year: "numeric",
      month: "short",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    });

    head.innerHTML = `
      <p class="print-head__badge">Screening grade — not for evacuation or dam-safety decisions</p>
      <h1>Dam-Breach Flood Screening Report</h1>
      <p class="print-head__sub">
        ${esc(dam.name || "—")} · ${esc(dam.state || "—")} ·
        ${esc(dam.dam_id || "—")} · SIH26161 · generated ${esc(generated)}
      </p>
      <dl class="print-head__meta">
        ${reportMetaPairs(payload)
          .map(([label, value]) => `<div><dt>${esc(label)}</dt><dd>${esc(String(value))}</dd></div>`)
          .join("")}
      </dl>`;

    const targets = (payload.targets || [])
      .map((t) => `${t.label} ${fmt(t.latitude, 4)}, ${fmt(t.longitude, 4)}`)
      .join(" · ");
    scenario.textContent =
      `${dam.name || "—"} (${dam.dam_id || "—"}): ` +
      `release level ${fmt(payload.inputs?.release_level_m, 1)} m MSL, ` +
      `release volume ${group(payload.inputs?.release_volume_mcm)} MCM, ` +
      `breach ${fmt(payload.inputs?.breach_head_m, 0)} m × ${fmt(payload.inputs?.breach_width_m, 0)} m, ` +
      `analysis radius ${fmt(payload.terrain?.radius_km, 0)} km, ` +
      `areas of interest: ${targets}. ` +
      `Terrain: ${payload.terrain?.provider || "—"}. ` +
      `Flooded area ${fmt(payload.summary?.estimated_flooded_area_km2, 2)} km², ` +
      `maximum depth ${fmt(payload.summary?.maximum_depth_m, 1)} m, ` +
      `peak discharge ${group(payload.summary?.peak_discharge_m3s)} m³/s. ` +
      `${payload.warning || ""}`;

    renderPrintFootprint(payload);
    renderPrintFoot(payload, generated);
  }

  /* ------------------------------------------------------ printed footprint */
  // The interactive map is dropped on paper, so the report draws its own plan of
  // the footprint from the same severity polygons that are exported as GeoJSON.
  const PRINT_BAND_FILL = ["#d9edf8", "#9fcde7", "#f8dc99", "#f0ad80", "#dc93a4"];
  const PRINT_BAND_STROKE = ["#4c9dc4", "#2f7fa8", "#a97c1f", "#b7682b", "#a1374d"];
  const PRINT_MAP_W = 720;
  const PRINT_MAP_PAD = 30;

  function geometryRings(geometry) {
    if (!geometry || !Array.isArray(geometry.coordinates)) return [];
    const polygons =
      geometry.type === "MultiPolygon" ? geometry.coordinates : [geometry.coordinates];
    const rings = [];
    polygons.forEach((polygon) =>
      (polygon || []).forEach((ring) => {
        if (Array.isArray(ring) && ring.length > 2) rings.push(ring);
      })
    );
    return rings;
  }

  function renderPrintFootprint(payload) {
    const container = $("#print-map");
    if (!container) return;
    container.innerHTML = "";

    const features = payload?.flood_geojson?.features || [];
    const dam = payload?.dam;
    if (!features.length || !dam || !Number.isFinite(Number(dam.latitude))) return;

    const latitude = Number(dam.latitude);
    const longitude = Number(dam.longitude);
    const kmPerDegLon = 111.32 * Math.max(Math.cos((latitude * Math.PI) / 180), 0.05);
    const project = (lon, lat) => [
      (lon - longitude) * kmPerDegLon,
      (latitude - lat) * 110.574,
    ];

    let minX = 0;
    let maxX = 0;
    let minY = 0;
    let maxY = 0;
    let north = -Infinity;
    let south = Infinity;
    let east = -Infinity;
    let west = Infinity;

    const projected = features.map((feature) => ({
      band: clamp(Number(feature?.properties?.band ?? 0), 0, PRINT_BAND_FILL.length - 1),
      rings: geometryRings(feature.geometry).map((ring) =>
        ring.map(([lon, lat]) => {
          north = Math.max(north, lat);
          south = Math.min(south, lat);
          east = Math.max(east, lon);
          west = Math.min(west, lon);
          const [x, y] = project(lon, lat);
          minX = Math.min(minX, x);
          maxX = Math.max(maxX, x);
          minY = Math.min(minY, y);
          maxY = Math.max(maxY, y);
          return [x, y];
        })
      ),
    }));

    const spanX = Math.max(maxX - minX, 0.5);
    const spanY = Math.max(maxY - minY, 0.5);
    // Match the plan's aspect ratio to the footprint, so a long reach is not
    // drawn as a sliver in the middle of a landscape box.
    const boxHeight = Math.round(clamp((PRINT_MAP_W * spanY) / spanX, 300, 880));
    const scale = Math.min(
      (PRINT_MAP_W - 2 * PRINT_MAP_PAD) / spanX,
      (boxHeight - 2 * PRINT_MAP_PAD) / spanY
    );
    const offsetX = (PRINT_MAP_W - spanX * scale) / 2 - minX * scale;
    const offsetY = (boxHeight - spanY * scale) / 2 - minY * scale;

    // Paper size of the plan, and how many viewBox units one millimetre is
    // worth — labels, markers and the scale bar are sized in millimetres so the
    // plan reads the same whether the footprint is long, wide or square.
    let planWidthMm = (92 * PRINT_MAP_W) / boxHeight;
    let planHeightMm = 92;
    if (planWidthMm > 182) {
      planHeightMm = (182 * boxHeight) / PRINT_MAP_W;
      planWidthMm = 182;
    }
    const perMm = boxHeight / planHeightMm;

    const toXy = ([x, y]) =>
      `${(offsetX + x * scale).toFixed(1)},${(offsetY + y * scale).toFixed(1)}`;
    const place = (lon, lat) => toXy(project(lon, lat)).split(",");

    const shapes = projected
      .map(({ band, rings }) => {
        if (!rings.length) return "";
        const path = rings.map((ring) => `M${ring.map(toXy).join("L")}Z`).join(" ");
        return (
          `<path d="${path}" fill="${PRINT_BAND_FILL[band]}" fill-opacity="0.62" ` +
          `stroke="${PRINT_BAND_STROKE[band]}" stroke-width="${(0.22 * perMm).toFixed(2)}" ` +
          `stroke-linejoin="round" fill-rule="evenodd"/>`
        );
      })
      .join("");

    const [damX, damY] = place(longitude, latitude);
    // Keep the name tag inside the plot: flip it to the left of the marker when
    // the dam sits near the right edge.
    const damFlip = Number(damX) > PRINT_MAP_W - PRINT_MAP_PAD - 24 * perMm;
    const damMarker =
      `<g transform="translate(${damX} ${damY}) scale(${perMm.toFixed(3)})">` +
      `<circle r="2.1" fill="#10161f"/>` +
      `<circle r="0.95" fill="#ffffff"/>` +
      (damFlip
        ? `<text class="print-map__label" x="-3.2" y="-1.3" text-anchor="end" ` +
          `style="font-size: 2.5px">${esc(dam.name || "Dam")}</text></g>`
        : `<text class="print-map__label" x="3.2" y="-1.3" ` +
          `style="font-size: 2.5px">${esc(dam.name || "Dam")}</text></g>`);

    const targetMarkers = (payload.targets || [])
      .filter(
        (target) =>
          Number.isFinite(Number(target.latitude)) && Number.isFinite(Number(target.longitude))
      )
      .map((target, index) => {
        const [cx, cy] = place(Number(target.longitude), Number(target.latitude));
        const wet = Boolean(target.inundated);
        return (
          `<g transform="translate(${cx} ${cy}) scale(${perMm.toFixed(3)})">` +
          `<circle r="2.6" fill="${wet ? "#b2113c" : "#ffffff"}" ` +
          `stroke="${wet ? "#7f0c2b" : "#35455a"}" stroke-width="0.3"/>` +
          `<text class="print-map__tick${wet ? " print-map__tick--light" : ""}" ` +
          `y="1.15" style="font-size: 3.2px">${index + 1}</text>` +
          `<text class="print-map__label" x="3.6" y="-2.4" ` +
          `style="font-size: 2.4px">${esc(target.label || `Area ${index + 1}`)}</text></g>`
        );
      })
      .join("");

    const niceLengths = [1, 2, 5, 10, 20, 50, 100, 200];
    const maxBarKm = ((PRINT_MAP_W - 2 * PRINT_MAP_PAD) * 0.34) / scale;
    const barKm = niceLengths.filter((value) => value <= maxBarKm).pop() || niceLengths[0];
    const barHalfMm = (barKm * scale) / perMm / 2;
    const barY = boxHeight - PRINT_MAP_PAD * 0.7;
    const scaleBar =
      `<g transform="translate(${PRINT_MAP_PAD} ${barY.toFixed(1)}) scale(${perMm.toFixed(3)})">` +
      `<rect width="${barHalfMm.toFixed(1)}" height="1.7" fill="#10161f"/>` +
      `<rect x="${barHalfMm.toFixed(1)}" width="${barHalfMm.toFixed(1)}" height="1.7" ` +
      `fill="#ffffff" stroke="#10161f" stroke-width="0.22"/>` +
      `<text class="print-map__label" y="-1.3" style="font-size: 2.6px">` +
      `${barKm} km</text></g>`;

    const northArrow =
      `<g transform="translate(${(PRINT_MAP_W - PRINT_MAP_PAD - 1.6 * perMm).toFixed(1)} ` +
      `${(PRINT_MAP_PAD + 9.5 * perMm).toFixed(1)}) scale(${perMm.toFixed(3)})">` +
      `<path d="M0,-4.4 L1.6,2.2 L0,0.9 L-1.6,2.2 Z" fill="#10161f"/>` +
      `<text class="print-map__northlabel" y="6" style="font-size: 2.8px">N</text></g>`;

    container.innerHTML = `
      <div class="print-map__body">
        <div class="print-map__canvas">
          <svg class="print-map__svg" width="${Math.round(planWidthMm)}mm"
               height="${Math.round(planHeightMm)}mm" viewBox="0 0 ${PRINT_MAP_W} ${boxHeight}"
               role="img" aria-label="Schematic of the modelled flood footprint, north up">
            <rect class="print-map__field" x="0.5" y="0.5"
                  width="${PRINT_MAP_W - 1}" height="${boxHeight - 1}" rx="3"/>
            ${shapes}
            ${damMarker}
            ${targetMarkers}
            ${scaleBar}
            ${northArrow}
          </svg>
        </div>
        <figcaption class="print-map__side">
          <span class="print-map__title">Inundation footprint · schematic plan, north up</span>
          <span class="print-map__note">
            Severity polygons over ${fmt((PRINT_MAP_W - 2 * PRINT_MAP_PAD) / scale, 1)} km across ·
            extent ${fmt(west, 4)}–${fmt(east, 4)}° E, ${fmt(south, 4)}–${fmt(north, 4)}° N ·
            the same polygons exported as GeoJSON.
          </span>
          <span class="print-map__legend">
            ${BAND_LABELS.map(
              (label, index) =>
                `<i style="--swatch: ${PRINT_BAND_FILL[index]}; ` +
                `--edge: ${PRINT_BAND_STROKE[index]}"></i>${esc(label)}`
            ).join("")}
          </span>
          <span class="print-map__key">
            <i class="print-map__key-dam"></i>Dam axis — origin of distances
          </span>
          <ul class="print-map__keylist">
            ${(payload?.targets || [])
              .filter(
                (target) =>
                  Number.isFinite(Number(target.latitude)) &&
                  Number.isFinite(Number(target.longitude))
              )
              .map(
                (target, index) =>
                  `<li><i class="print-map__key-dot${target.inundated ? "" : " is-dry"}">` +
                  `${index + 1}</i>${esc(target.label || `Area ${index + 1}`)} — ` +
                  `${fmt(target.distance_from_dam_km, 1)} km downstream, ` +
                  `${target.inundated ? `${esc(target.severity || "wet")} · ${fmt(target.flood_depth_m, 2)} m deep` : "dry"}</li>`
              )
              .join("")}
          </ul>
        </figcaption>
      </div>`;
  }

  function renderPrintFoot(payload, generated) {
    const container = $("#print-foot");
    if (!container) return;

    const terrain = payload?.terrain || {};
    const check = terrain.vertical_check;
    const flagged = (payload?.notes || []).filter((note) => note.level === "warning").length;

    container.innerHTML = `
      <p class="print-foot__warning">${esc(
        payload?.warning || "Screening approximation only."
      )}</p>
      <p class="print-foot__meta">
        Terrain ${esc(terrain.provider || "—")}${
          terrain.resolution_m ? ` · ${fmt(terrain.resolution_m, 1)} m/px` : ""
        } · vertical check ${
          check && check.mean_abs_delta_m != null
            ? `${fmt(check.mean_abs_delta_m, 2)} m mean |Δ| against ${esc(check.reference || "—")}`
            : "not run"
        } · exposure © OpenStreetMap contributors (ODbL)${
          flagged ? ` · ${flagged} flagged caveat${flagged === 1 ? "" : "s"} in this run` : ""
        } · SIH26161 · ${esc(generated)}
      </p>`;
  }

  function summaryText(payload) {
    const stats = payload.summary?.depth_stats || {};
    const lines = [
      `SIH26161 DAM-BREACH FLOOD SCREENING — ${payload.dam?.name || ""} (${payload.dam?.dam_id || ""})`,
      "",
      ...reportMetaPairs(payload).map(([label, value]) => `${label}: ${value}`),
      `Flow direction (E, N): ${(payload.inputs?.flow_direction_east_north || [])
        .map((v) => fmt(v, 3))
        .join(", ")}`,
      `Processing time: ${fmt(payload.processing_ms, 0)} ms`,
      "",
      "AREAS OF INTEREST",
      ...(payload.targets || []).map(
        (t) =>
          `- ${t.label} (${fmt(t.latitude, 4)}, ${fmt(t.longitude, 4)}) — ` +
          `${fmt(t.distance_from_dam_km, 1)} km downstream, ground ${fmt(t.ground_elevation_m, 1)} m, ` +
          `depth ${t.inundated ? `${fmt(t.flood_depth_m, 2)} m (${t.severity})` : "dry"}`
      ),
    ];

    if (payload.exposure?.available && payload.exposure.groups?.length) {
      lines.push("", "ASSET EXPOSURE (OpenStreetMap)");
      payload.exposure.groups.forEach((g) => {
        lines.push(`- ${g.label}: ${g.inundated} of ${g.total} inside the flood`);
      });
    }

    if (payload.notes?.length) {
      lines.push("", "MODEL CAVEATS");
      payload.notes.forEach((n) => lines.push(`- [${n.level}] ${n.text}`));
    }

    lines.push("", payload.warning || "");
    return lines.join("\n");
  }

  async function copySummary() {
    if (!state.payload) return;
    const text = summaryText(state.payload);
    try {
      await navigator.clipboard.writeText(text);
      toast("Summary copied to the clipboard.", "success");
    } catch {
      const area = document.createElement("textarea");
      area.value = text;
      area.setAttribute("readonly", "");
      area.style.position = "fixed";
      area.style.top = "-1000px";
      document.body.appendChild(area);
      area.select();
      const ok = document.execCommand("copy");
      document.body.removeChild(area);
      toast(ok ? "Summary copied." : "Copy failed — select the text manually.", ok ? "success" : "error");
    }
  }

  function printReport() {
    if (!state.payload) return;
    renderPrintHead(state.payload);
    window.print();
  }

  function renderTargetStatuses(targets = []) {
    targets.forEach((target, index) => {
      const fieldset = $(`.target[data-target="${index + 1}"]`);
      const status = $(`[data-status="${index + 1}"]`);
      if (!fieldset || !status) return;

      fieldset.classList.toggle("is-flooded", Boolean(target.inundated));
      fieldset.classList.toggle("is-dry", !target.inundated);

      if (!target.inside_grid) {
        status.textContent = "Outside the modelled grid";
        return;
      }
      status.textContent = target.inundated
        ? `${fmt(target.flood_depth_m)} m depth · ${target.severity}`
        : `Dry · ${fmt(target.distance_from_dam_km, 1)} km from the dam`;
    });
  }

  function flagDamSnap(payload) {
    const note = $("#snap-note");
    if (!note) return;
    const distance = Number(payload?.inputs?.dam_snap_distance_km ?? 0);
    const snapped = Boolean(payload?.inputs?.dam_snapped_to_nearest_cell);

    if (!snapped || distance <= SNAP_WARN_KM) {
      note.hidden = true;
      return;
    }

    note.hidden = false;
    note.textContent =
      `The dam falls outside the terrain window, so the model started ${fmt(distance, 1)} km ` +
      "from it. Treat this run as unreliable — widen the radius or check the coordinates.";
    toast("Dam is outside the terrain window — results are unreliable.", "error", 7000);
  }

  function drawFlood(payload) {
    if (!mapReady()) return;

    if (floodLayer) map.removeLayer(floodLayer);
    if (assetLayer) {
      map.removeLayer(assetLayer);
      assetLayer = null;
    }

    floodLayer = L.geoJSON(payload.flood_geojson, {
      style: (feature) => {
        const band = Number(feature?.properties?.band ?? 0);
        return {
          color: BAND_COLORS[band] ?? "#38bdf8",
          weight: 1,
          fillColor: BAND_COLORS[band] ?? "#38bdf8",
          fillOpacity: 0.35,
        };
      },
      onEachFeature: (feature, layer) => {
        const band = Number(feature?.properties?.band ?? 0);
        const label = state.labels[band] ?? feature?.properties?.severity ?? "flood";
        layer.bindTooltip(`Flood depth ${label}`, { sticky: true });
      },
    }).addTo(map);

    if (payload.exposure?.available && Array.isArray(payload.exposure.assets)) {
      const markers = payload.exposure.assets
        .filter((asset) => asset.inundated)
        .slice(0, 60)
        .map((asset) =>
          L.circleMarker([asset.latitude, asset.longitude], {
            radius: 4,
            color: "#fda4af",
            weight: 1,
            fillColor: asset.inundated ? "#f43f5e" : "#94a3b8",
            fillOpacity: 0.9,
          }).bindPopup(
            popup(asset.name, [
              ["Category", asset.group_label],
              ["Depth", `${fmt(asset.flood_depth_m, 1)} m`],
            ])
          )
        );
      if (markers.length) {
        assetLayer = L.layerGroup(markers).addTo(map);
      }
    }

    $("#legend").hidden = false;
    $("#legend-note").textContent = payload.synthetic
      ? "Synthetic footprint (demo mode)"
      : `${payload.flood_geojson.features.length} severity zones`;

    updateDamMarker();
    updateTargetMarker(1);
    updateTargetMarker(2);

    const bounds = floodLayer.getBounds();
    if (bounds.isValid()) {
      map.fitBounds(bounds, { padding: [40, 40], maxZoom: 12 });
    } else if (state.dam) {
      map.setView([state.dam.latitude, state.dam.longitude], 10);
    }

    // A new result invalidates the 3-D valley: rebuild it on next open.
    valleyInvalidate();
  }

  /* ============================================================ 3-D valley
     Real Three.js scene over the modelled grids (see VALLEY block above).
     ========================================================= */

  // Decode payload.grid3d into plain Float32 grids (metres).
  function valleyDecode(payload) {
    const block = payload?.grid3d;
    if (!block || !Array.isArray(block.elevation_decim) || !Array.isArray(block.depth_decim)) {
      return null;
    }
    const rows = Number(block.rows);
    const cols = Number(block.cols);
    if (!Number.isFinite(rows) || !Number.isFinite(cols) || rows < 2 || cols < 2) return null;
    if (block.elevation_decim.length < rows * cols || block.depth_decim.length < rows * cols) {
      return null;
    }
    const scale = Number(block.scale) || 10;
    const nodata = block.nodata ?? -999999;
    const elevation = new Float32Array(rows * cols);
    const depth = new Float32Array(rows * cols);
    let minElev = Infinity;
    let maxElev = -Infinity;
    let maxDepth = 0;
    for (let index = 0; index < rows * cols; index += 1) {
      const rawElev = block.elevation_decim[index];
      const rawDepth = block.depth_decim[index];
      const elev = rawElev === nodata ? NaN : Number(rawElev) / scale;
      const wet = Number.isFinite(Number(rawDepth)) && Number(rawDepth) > 0 ? Number(rawDepth) / scale : 0;
      elevation[index] = elev;
      depth[index] = wet;
      if (Number.isFinite(elev)) {
        if (elev < minElev) minElev = elev;
        if (elev > maxElev) maxElev = elev;
      }
      if (wet > maxDepth) maxDepth = wet;
    }
    if (!Number.isFinite(minElev)) return null;
    return {
      rows,
      cols,
      cellX: Number(block.cell_size_x_m) || 100,
      cellY: Number(block.cell_size_y_m) || 100,
      damRow: clamp(Number(block.dam_row) || 0, 0, rows - 1),
      damCol: clamp(Number(block.dam_col) || 0, 0, cols - 1),
      damHeightM: Number.isFinite(Number(block.dam_height_m)) ? Number(block.dam_height_m) : null,
      releaseLevelM: Number(block.release_level_m),
      poolLevelM: Number(block.pool_level_m),
      elevation,
      depth,
      minElev,
      maxElev,
      maxDepth,
      synthetic: Boolean(payload.synthetic),
    };
  }

  // Front curve shared with the 2-D replay.
  function valleyFront(payload) {
    const summary = payload?.summary || {};
    const inputs = payload?.inputs || {};
    const peak = Number(summary.peak_discharge_m3s) || 0;
    const head = Math.max(Number(inputs.breach_head_m) || 0, 0.1);
    const width = Math.max(Number(inputs.breach_width_m) || head, 1);
    const v0 = clamp(peak > 0 ? peak / Math.max(head * width, 1) : 4, 0.5, 25);
    const grid = VALLEY.grid;
    let reachM = 0;
    if (grid) {
      for (let row = 0; row < grid.rows; row += 1) {
        for (let col = 0; col < grid.cols; col += 1) {
          if (grid.depth[row * grid.cols + col] > 0.05) {
            const east = (col - grid.damCol) * grid.cellX;
            const north = (grid.damRow - row) * grid.cellY;
            const dist = Math.hypot(east, north);
            if (dist > reachM) reachM = dist;
          }
        }
      }
    }
    if (!(reachM > 500)) reachM = 10000;
    const tauS = reachM / Math.max(v0, 1e-6);
    const endS = tauS * Math.log(1 / (1 - SIM_HORIZON));
    return {
      v0,
      tauS,
      endS,
      distanceM: (seconds) => reachM * (1 - Math.exp(-seconds / tauS)),
    };
  }

  function valleySetView(view) {
    const two = view !== "3d";
    $$("[data-mapview]").forEach((button) =>
      button.classList.toggle("is-active", (button.dataset.mapview === "3d") !== two)
    );
    if (two) {
      valleyClose();
      return;
    }
    valleyOpen();
  }

  function valleyInvalidate() {
    VALLEY.grid = null;
    VALLEY.damInfo = null;
    VALLEY.hudKey = "";
    VALLEY.clockS = 0;
    if (!VALLEY.open) return;
    valleyBuildScene(true);
  }

  async function valleyThree() {
    if (VALLEY.three) return VALLEY.three;
    if (!VALLEY.loading) {
      VALLEY.loading = import(VALLEY_CDN).catch((error) => {
        VALLEY.loading = null;
        throw error;
      });
    }
    VALLEY.three = await VALLEY.loading;
    return VALLEY.three;
  }

  async function valleyOpen() {
    if (!state.payload) {
      toast("Run a screening first — the 3-D valley renders its result.", "warn");
      valleySetView("2d");
      return;
    }
    const grid = valleyDecode(state.payload);
    if (!grid) {
      toast("This result carries no 3-D grid yet — re-run the screening.", "warn");
      valleySetView("2d");
      return;
    }
    VALLEY.grid = grid;
    VALLEY.damInfo = {
      name: state.payload?.dam?.name || "Dam",
      state: state.payload?.dam?.state || "",
      heightM: grid.damHeightM,
      headM: Number(state.payload?.inputs?.breach_head_m) || null,
      poolM: grid.poolLevelM,
    };
    VALLEY.open = true;
    $("#valley").hidden = false;
    if (sim.open && sim.playing) simSetPlaying(false);
    try {
      await valleyThree();
    } catch (error) {
      VALLEY.open = false;
      $("#valley").hidden = true;
      valleySetView("2d");
      toast("3-D library (three.js CDN) is unreachable — check the connection.", "error", 6000);
      return;
    }
    valleyBuildScene(false);
    valleyHooks();
    VALLEY.lastMs = 0;
    if (!VALLEY.raf) VALLEY.raf = requestAnimationFrame(valleyLoop);
    valleyHud(true);
  }

  function valleyClose() {
    if (!VALLEY.open && $("#valley")?.hidden !== false) return;
    VALLEY.open = false;
    VALLEY.playing = false;
    if (VALLEY.raf) cancelAnimationFrame(VALLEY.raf);
    VALLEY.raf = 0;
    const panel = $("#valley");
    if (panel) panel.hidden = true;
    if (VALLEY.renderer) {
      try {
        VALLEY.renderer.dispose();
      } catch (error) { /* best effort */ }
      VALLEY.renderer = null;
    }
    VALLEY.scene = null;
    VALLEY.camera = null;
    VALLEY.terrain = null;
    VALLEY.water = null;
    VALLEY.dam = null;
    VALLEY.targets = [];
    VALLEY.rays = null;
    $$("[data-mapview]").forEach((button) =>
      button.classList.toggle("is-active", button.dataset.mapview !== "3d")
    );
  }

  function valleyOrbitDefaults() {
    VALLEY.orbit = { theta: 0.85, phi: 1.02, radius: 1.6, target: [0, 0, 0] };
  }

  // Build (or rebuild, preserving the camera) the scene from VALLEY.grid.
  function valleyBuildScene(keepCamera) {
    const THREE = VALLEY.three;
    const grid = VALLEY.grid;
    const canvas = $("#valley-canvas");
    if (!THREE || !grid || !canvas) return;
    const savedOrbit = keepCamera ? { ...VALLEY.orbit } : null;
    const width = canvas.clientWidth || canvas.parentElement.clientWidth || 600;
    const height = canvas.clientHeight || canvas.parentElement.clientHeight || 420;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (!VALLEY.renderer) {
      VALLEY.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    }
    const renderer = VALLEY.renderer;
    renderer.setPixelRatio(dpr);
    renderer.setSize(width, height, false);
    VALLEY.size = { w: width, h: height };
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x060c15);
    scene.fog = new THREE.Fog(0x060c15, 2.6, 5.2);
    VALLEY.scene = scene;
    const camera = new THREE.PerspectiveCamera(46, width / Math.max(height, 1), 0.01, 50);
    VALLEY.camera = camera;
    const key = new THREE.DirectionalLight(0xd8ecff, 1.5);
    key.position.set(-0.8, 1.4, 0.6);
    scene.add(key);
    const rim = new THREE.DirectionalLight(0x38bdf8, 0.55);
    rim.position.set(1.2, 0.5, -1.0);
    scene.add(rim);
    scene.add(new THREE.AmbientLight(0x8aa3c0, 0.5));
    // World frame: X = east, Z = south (row+), Y = up.
    const spanX = grid.cols * grid.cellX;
    const spanZ = grid.rows * grid.cellY;
    const span = Math.max(spanX, spanZ, 1);
    const relief = Math.max(grid.maxElev - grid.minElev, 1);
    const mid = (grid.minElev + grid.maxElev) / 2;
    const toX = (col) => (col / (grid.cols - 1) - 0.5) * (spanX / span);
    const toZ = (row) => (row / (grid.rows - 1) - 0.5) * (spanZ / span);
    const toY = (elevM) => ((elevM - mid) / span) * VALLEY.lift;
    VALLEY.frame = { span, mid, toX, toZ, toY };
    const terrainGeo = new THREE.PlaneGeometry(1, 1, grid.cols - 1, grid.rows - 1);
    terrainGeo.rotateX(-Math.PI / 2);
    const positions = terrainGeo.attributes.position;
    const colours = new Float32Array(positions.count * 3);
    for (let row = 0; row < grid.rows; row += 1) {
      for (let col = 0; col < grid.cols; col += 1) {
        const vertex = row * grid.cols + col;
        const elev = grid.elevation[vertex];
        const finite = Number.isFinite(elev);
        positions.setXYZ(vertex, toX(col), finite ? toY(elev) : toY(grid.minElev - relief * 0.05), toZ(row));
        const east = grid.elevation[row * grid.cols + Math.min(col + 1, grid.cols - 1)];
        const west = grid.elevation[row * grid.cols + Math.max(col - 1, 0)];
        const south = grid.elevation[Math.min(row + 1, grid.rows - 1) * grid.cols + col];
        const north = grid.elevation[Math.max(row - 1, 0) * grid.cols + col];
        let shade = 1;
        if (finite && [east, west, south, north].every(Number.isFinite)) {
          const dx = (east - west) / Math.max(grid.cellX, 1);
          const dz = (south - north) / Math.max(grid.cellY, 1);
          const inv = 1 / Math.max(Math.hypot(dx, dz, 1 / Math.max(VALLEY.lift, 1e-6)), 1e-6);
          shade = clamp((-dx * -0.55 + (1 / Math.max(VALLEY.lift, 1e-6)) * 0.8 - dz * 0.35) * inv, 0.12, 1.15);
        }
        const t = clamp(finite ? (elev - grid.minElev) / relief : 0, 0, 1);
        let r; let g; let b;
        if (t < 0.45) {
          const mix = t / 0.45;
          r = 24 + 62 * mix; g = 52 + 52 * mix; b = 66 + 56 * mix;
        } else if (t < 0.8) {
          const mix = (t - 0.45) / 0.35;
          r = 86 + 62 * mix; g = 104 + 46 * mix; b = 122 + 36 * mix;
        } else {
          const mix = (t - 0.8) / 0.2;
          r = 148 + 78 * mix; g = 150 + 82 * mix; b = 158 + 82 * mix;
        }
        if (grid.depth[vertex] > 0.05) { r *= 0.75; g *= 0.92; }
        colours[vertex * 3] = clamp((r / 255) * (0.55 + 0.45 * shade), 0, 1);
        colours[vertex * 3 + 1] = clamp((g / 255) * (0.55 + 0.45 * shade), 0, 1);
        colours[vertex * 3 + 2] = clamp((b / 255) * (0.55 + 0.45 * shade), 0, 1);
      }
    }
    terrainGeo.setAttribute("color", new THREE.BufferAttribute(colours, 3));
    terrainGeo.computeVertexNormals();
    VALLEY.terrain = new THREE.Mesh(
      terrainGeo,
      new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.95, metalness: 0.02 })
    );
    scene.add(VALLEY.terrain);
    // Water surface over ground + solved depth (revealed per-frame below).
    const waterGeo = new THREE.PlaneGeometry(1, 1, grid.cols - 1, grid.rows - 1);
    waterGeo.rotateX(-Math.PI / 2);
    waterGeo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(waterGeo.attributes.position.count * 3), 3));
    VALLEY.water = new THREE.Mesh(
      waterGeo,
      new THREE.MeshStandardMaterial({ vertexColors: true, transparent: true, opacity: 0.82, roughness: 0.25, metalness: 0.1 })
    );
    VALLEY.water.position.y = 0.002;
    scene.add(VALLEY.water);
    // Dam wall across the valley at the dam cell.
    const damGroup = new THREE.Group();
    const wallH = Math.max(VALLEY.damInfo?.heightM || VALLEY.damInfo?.headM || 30, 8);
    const damGround = grid.elevation[grid.damRow * grid.cols + grid.damCol];
    const base = Number.isFinite(damGround) ? damGround : grid.minElev;
    const wallTopY = toY(base + wallH);
    const wallBaseY = toY(base - wallH * 0.4);
    const crestLen = span * 0.16;
    const wallGeo = new THREE.BoxGeometry(crestLen, Math.max(wallTopY - wallBaseY, 0.02), span * 0.022, 8, 1, 1);
    const wallPos = wallGeo.attributes.position;
    for (let vertex = 0; vertex < wallPos.count; vertex += 1) {
      const px = wallPos.getX(vertex);
      wallPos.setZ(vertex, wallPos.getZ(vertex) + Math.pow(px / Math.max(crestLen, 1e-6), 2) * span * 0.03);
    }
    wallGeo.computeVertexNormals();
    damGroup.add(new THREE.Mesh(
      wallGeo,
      new THREE.MeshStandardMaterial({ color: 0x9aa7b8, roughness: 0.6, metalness: 0.25 })
    ));
    const crest = new THREE.Mesh(
      new THREE.BoxGeometry(crestLen * 1.02, 0.008, span * 0.03),
      new THREE.MeshStandardMaterial({ color: 0xe6f0f9, roughness: 0.5 })
    );
    crest.position.y = (wallTopY - wallBaseY) / 2;
    damGroup.add(crest);
    const notch = new THREE.Mesh(
      new THREE.BoxGeometry(span * 0.03, Math.max(wallTopY - wallBaseY, 0.02) * 0.9, span * 0.024),
      new THREE.MeshBasicMaterial({ color: 0x04080f })
    );
    notch.position.y = -((wallTopY - wallBaseY) * 0.02);
    damGroup.add(notch);
    VALLEY.damNotch = notch;
    damGroup.position.set(toX(grid.damCol), (wallTopY + wallBaseY) / 2, toZ(grid.damRow));
    damGroup.rotation.y = Math.PI / 2;
    scene.add(damGroup);
    VALLEY.dam = damGroup;
    // Target pins (red = wet, blue = dry).
    VALLEY.targets = [];
    const targets = Array.isArray(state.payload?.targets) ? state.payload.targets : [];
    const damLat = Number(state.payload?.dam?.latitude);
    const damLon = Number(state.payload?.dam?.longitude);
    const kmPerDegLon = 111.32 * Math.max(Math.cos((damLat * Math.PI) / 180), 0.05);
    targets
      .filter((target) => Number.isFinite(Number(target.latitude)) && Number.isFinite(Number(target.longitude)))
      .slice(0, 8)
      .forEach((target) => {
        const eastM = (Number(target.longitude) - damLon) * kmPerDegLon * 1000;
        const southM = (damLat - Number(target.latitude)) * 110574;
        const col = grid.damCol + eastM / Math.max(grid.cellX, 1);
        const row = grid.damRow + southM / Math.max(grid.cellY, 1);
        if (row < 0 || row >= grid.rows - 1 || col < 0 || col >= grid.cols - 1) return;
        const r0 = clamp(Math.floor(row), 0, grid.rows - 1);
        const c0 = clamp(Math.floor(col), 0, grid.cols - 1);
        const elev = grid.elevation[r0 * grid.cols + c0];
        const wet = grid.depth[r0 * grid.cols + c0];
        const pin = new THREE.Group();
        const stemH = span * 0.05;
        const stem = new THREE.Mesh(
          new THREE.CylinderGeometry(span * 0.0022, span * 0.0022, stemH, 8),
          new THREE.MeshBasicMaterial({ color: 0xe6f0f9 })
        );
        stem.position.y = stemH / 2;
        const head = new THREE.Mesh(
          new THREE.SphereGeometry(span * 0.008, 16, 12),
          new THREE.MeshBasicMaterial({ color: wet > 0.05 ? 0xf43f5e : 0x38bdf8 })
        );
        head.position.y = stemH;
        pin.add(stem);
        pin.add(head);
        pin.position.set(toX(col), Number.isFinite(elev) ? toY(elev) : 0, toZ(row));
        scene.add(pin);
        VALLEY.targets.push(pin);
      });
    // Arrival-time contour rings from the same front curve.
    const raysGroup = new THREE.Group();
    const front = valleyFront(state.payload);
    [0.25, 0.5, 0.75, 1.0].forEach((fraction, index) => {
      const radiusM = front.distanceM(front.endS) * fraction;
      const ringGeo = new THREE.RingGeometry(Math.max(radiusM / span - span * 0.0012, 0.001), Math.max(radiusM / span, 0.002), 96);
      ringGeo.rotateX(-Math.PI / 2);
      const ring = new THREE.Mesh(
        ringGeo,
        new THREE.MeshBasicMaterial({ color: index === 3 ? 0xfbbf24 : 0x22d3ee, transparent: true, opacity: 0.35, side: THREE.DoubleSide, depthWrite: false })
      );
      ring.position.set(toX(grid.damCol), toY(grid.maxElev) + 0.01 + index * 0.002, toZ(grid.damRow));
      raysGroup.add(ring);
    });
    scene.add(raysGroup);
    VALLEY.rays = raysGroup;
    VALLEY.front = front;
    if (savedOrbit) VALLEY.orbit = savedOrbit;
    else valleyOrbitDefaults();
    valleyApplyOrbit();
    valleyUpdateWater(true);
    valleyLegend();
    valleyHud(true);
  }

  // Per-frame water update: reveal cells the front has reached.
  function valleyUpdateWater(force) {
    const THREE = VALLEY.three;
    const grid = VALLEY.grid;
    if (!THREE || !grid || !VALLEY.water) return;
    const front = VALLEY.front || valleyFront(state.payload);
    VALLEY.front = front;
    const frontM = front.distanceM(VALLEY.clockS);
    const positions = VALLEY.water.geometry.attributes.position;
    const colourAttr = VALLEY.water.geometry.attributes.color;
    const frame = VALLEY.frame;
    let shown = 0;
    for (let row = 0; row < grid.rows; row += 1) {
      for (let col = 0; col < grid.cols; col += 1) {
        const index = row * grid.cols + col;
        const ground = grid.elevation[index];
        const solved = grid.depth[index];
        const east = (col - grid.damCol) * grid.cellX;
        const north = (grid.damRow - row) * grid.cellY;
        const distM = Math.hypot(east, north);
        const arrived = distM <= frontM + 1;
        const shownDepth = arrived ? solved * VALLEY.breach : 0;
        if (shownDepth > 0.05) shown += 1;
        const surface = Number.isFinite(ground) ? ground + shownDepth * VALLEY.level : frame.mid;
        positions.setXYZ(index, frame.toX(col), frame.toY(surface) + 0.002, frame.toZ(row));
        const rgb = valleyRamp(shownDepth, Math.max(grid.maxDepth, 0.5));
        const dim = arrived ? 1 : 0;
        colourAttr.setXYZ(index, (rgb[0] / 255) * dim, (rgb[1] / 255) * dim, (rgb[2] / 255) * dim);
      }
    }
    positions.needsUpdate = true;
    colourAttr.needsUpdate = true;
    VALLEY.water.geometry.computeVertexNormals();
    VALLEY.water.visible = shown > 0;
    if (VALLEY.damNotch) {
      VALLEY.damNotch.scale.x = clamp(1 - VALLEY.breach + 0.15, 0.12, 1);
      VALLEY.damNotch.visible = VALLEY.breach < 0.99;
    }
    VALLEY.shown = shown;
    void force;
  }

  function valleyLegend() {
    const legend = $("#valley-legend");
    if (!legend) return;
    legend.innerHTML =
      `<span><i style="background:#22d3ee"></i>shallow</span>` +
      `<span><i style="background:#fbbf24"></i>deep</span>` +
      `<span><i style="background:#f43f5e"></i>abyss</span>` +
      `<span><i style="background:#9aa7b8"></i>dam</span>`;
  }

  function valleyHud(force) {
    const hud = $("#valley-hud");
    if (!hud || !VALLEY.grid || !VALLEY.front) return;
    const key = [Math.round(VALLEY.clockS), Math.round(VALLEY.breach * 100), Math.round(VALLEY.level * 100)].join("|");
    if (!force && key === VALLEY.hudKey) return;
    VALLEY.hudKey = key;
    const grid = VALLEY.grid;
    const front = VALLEY.front;
    const frontKm = front.distanceM(VALLEY.clockS) / 1000;
    const reachKm = front.distanceM(front.endS) / 1000;
    const info = VALLEY.damInfo || {};
    const wallM = info.heightM != null ? `${fmt(info.heightM, 0)} m wall` : info.headM != null ? `${fmt(info.headM, 1)} m head` : "wall";
    hud.innerHTML =
      `<strong>${esc(info.name || "Dam")}</strong>${info.state ? ` · ${esc(info.state)}` : ""}<br>` +
      `3-D valley over the modelled DEM — <code>${grid.cols}×${grid.rows}</code> cells, ` +
      `dam <code>${wallM}</code>, pool <code>${fmt(grid.poolLevelM, 1)} m</code>.<br>` +
      `Front <code>${fmt(frontKm, 1)} / ${fmt(reachKm, 1)} km</code> · ` +
      `T+<code>${simTime(VALLEY.clockS)}</code> · ` +
      `water <code>${Math.round(VALLEY.breach * 100)}%</code>.`;
    const note = $("#valley-note");
    if (note) {
      note.textContent = "Drag to orbit · wheel to zoom · right-drag to pan. Sliders rescale the render only — report numbers never change.";
    }
    const title = $("#valley-title");
    if (title) title.textContent = `3-D valley — ${info.name || "dam"}`;
  }

  function valleyApplyOrbit() {
    const camera = VALLEY.camera;
    if (!camera) return;
    const orbit = VALLEY.orbit;
    orbit.phi = clamp(orbit.phi, 0.15, 1.45);
    orbit.radius = clamp(orbit.radius, 0.6, 4.5);
    const sp = Math.sin(orbit.phi);
    camera.position.set(
      orbit.target[0] + orbit.radius * sp * Math.sin(orbit.theta),
      orbit.target[1] + orbit.radius * Math.cos(orbit.phi),
      orbit.target[2] + orbit.radius * sp * Math.cos(orbit.theta)
    );
    camera.lookAt(orbit.target[0], orbit.target[1], orbit.target[2]);
  }

  function valleyLoop(nowMs) {
    if (!VALLEY.open) {
      VALLEY.raf = 0;
      return;
    }
    const dt = VALLEY.lastMs ? Math.min((nowMs - VALLEY.lastMs) / 1000, 0.1) : 0;
    VALLEY.lastMs = nowMs;
    if (VALLEY.playing && VALLEY.front) {
      VALLEY.clockS += dt * (VALLEY.front.endS / SIM_RUN_SECONDS);
      if (VALLEY.clockS >= VALLEY.front.endS) {
        VALLEY.clockS = VALLEY.front.endS;
        VALLEY.playing = false;
        const button = $("#valley-play");
        if (button) {
          button.textContent = "Play";
          button.setAttribute("aria-pressed", "false");
        }
      }
      valleyUpdateWater(false);
      valleyHud(false);
    }
    if (VALLEY.spin) {
      VALLEY.orbit.theta += dt * 0.25;
      valleyApplyOrbit();
    }
    if (VALLEY.renderer && VALLEY.scene && VALLEY.camera) {
      const shimmer = 0.78 + 0.06 * Math.sin(nowMs / 700);
      if (VALLEY.water) VALLEY.water.material.opacity = shimmer;
      VALLEY.renderer.render(VALLEY.scene, VALLEY.camera);
    }
    VALLEY.raf = requestAnimationFrame(valleyLoop);
  }

  function valleyHooks() {
    if (VALLEY.hooks) return;
    VALLEY.hooks = true;
    const canvas = $("#valley-canvas");
    let dragging = null;
    const pointAt = (event) => {
      const rect = canvas.getBoundingClientRect();
      return { x: event.clientX - rect.left, y: event.clientY - rect.top };
    };
    canvas.addEventListener("pointerdown", (event) => {
      canvas.setPointerCapture(event.pointerId);
      dragging = { ...pointAt(event), button: event.button, shift: event.shiftKey };
    });
    canvas.addEventListener("pointermove", (event) => {
      if (!dragging) return;
      const point = pointAt(event);
      const dx = point.x - dragging.x;
      const dy = point.y - dragging.y;
      dragging = { ...point, button: dragging.button, shift: dragging.shift };
      if (dragging.button === 2 || dragging.shift) {
        const scale = VALLEY.orbit.radius * 0.0016;
        const cosT = Math.cos(VALLEY.orbit.theta);
        const sinT = Math.sin(VALLEY.orbit.theta);
        VALLEY.orbit.target[0] -= dx * cosT * scale;
        VALLEY.orbit.target[2] += dx * sinT * scale;
        VALLEY.orbit.target[1] += dy * scale;
        VALLEY.orbit.target[1] = clamp(VALLEY.orbit.target[1], -0.8, 0.8);
      } else {
        VALLEY.orbit.theta -= dx * 0.005;
        VALLEY.orbit.phi -= dy * 0.004;
      }
      valleyApplyOrbit();
    });
    const endDrag = () => { dragging = null; };
    canvas.addEventListener("pointerup", endDrag);
    canvas.addEventListener("pointercancel", endDrag);
    canvas.addEventListener("contextmenu", (event) => event.preventDefault());
    canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      VALLEY.orbit.radius *= event.deltaY > 0 ? 1.09 : 0.92;
      valleyApplyOrbit();
    }, { passive: false });
    window.addEventListener("resize", () => {
      if (!VALLEY.open || !VALLEY.renderer || !VALLEY.camera) return;
      const width = canvas.clientWidth || VALLEY.size.w || 600;
      const height = canvas.clientHeight || VALLEY.size.h || 420;
      VALLEY.renderer.setSize(width, height, false);
      VALLEY.camera.aspect = width / Math.max(height, 1);
      VALLEY.camera.updateProjectionMatrix();
    });
    $$("[data-mapview]").forEach((button) =>
      button.addEventListener("click", () => valleySetView(button.dataset.mapview))
    );
    $("#valley-close").addEventListener("click", () => valleySetView("2d"));
    $("#valley-reset").addEventListener("click", () => {
      valleyOrbitDefaults();
      valleyApplyOrbit();
    });
    $("#valley-play").addEventListener("click", () => {
      VALLEY.playing = !VALLEY.playing;
      VALLEY.lastMs = 0;
      if (VALLEY.playing && VALLEY.front && VALLEY.clockS >= VALLEY.front.endS) VALLEY.clockS = 0;
      const button = $("#valley-play");
      button.textContent = VALLEY.playing ? "Pause" : "Play";
      button.setAttribute("aria-pressed", String(VALLEY.playing));
    });
    $("#valley-spin").addEventListener("click", () => {
      VALLEY.spin = !VALLEY.spin;
      const button = $("#valley-spin");
      button.textContent = VALLEY.spin ? "Orbit: on" : "Orbit: off";
      button.setAttribute("aria-pressed", String(VALLEY.spin));
    });
    const breach = $("#valley-breach");
    breach.addEventListener("input", () => {
      VALLEY.breach = clamp(Number(breach.value) / 100, 0, 1);
      $("#valley-breach-out").textContent = `${breach.value}%`;
      valleyUpdateWater(true);
      valleyHud(true);
    });
    const level = $("#valley-level");
    level.addEventListener("input", () => {
      VALLEY.level = clamp(Number(level.value) / 100, 0, 1);
      const out = $("#valley-level-out");
      if (out) out.textContent = level.value === "100" ? "full" : `${level.value}%`;
      valleyUpdateWater(true);
      valleyHud(true);
    });
    const lift = $("#valley-lift");
    lift.addEventListener("input", () => {
      VALLEY.lift = clamp(Number(lift.value) / 100, 0.5, 4);
      const out = $("#valley-lift-out");
      if (out) out.textContent = `${VALLEY.lift.toFixed(1)}x`;
      valleyBuildScene(true);
    });
  }

  /* Animation replay (2-D canvas over the Leaflet map). */

  const SIM_SCENARIOS = [
    {
      id: "surge",
      label: "1 · Breach surge",
      title: "Dam breach and hydrodynamic surge",
      blurb: "The front leaves the dam at the breach celerity and decelerates as it spreads.",
    },
    {
      id: "velocity",
      label: "2 · Velocity heatmap",
      title: "Water velocity and impact force",
      blurb: "Colour is front celerity where the wave arrived; force is ½ρv² on that velocity.",
    },
    {
      id: "inundation",
      label: "3 · Inundation",
      title: "Inundation perimeter and area coverage",
      blurb: "Depth-class polygons fill in behind the front; the ticker sums the rasterised footprint.",
    },
    {
      id: "timeline",
      label: "4 · Timeline & ETA",
      title: "Timeline and location propagation",
      blurb: "Arrival times for the areas of interest, read off the same front curve.",
    },
  ];

  const SIM_HORIZON = 0.995; // the front stops at 99.5 % of the modelled reach
  const SIM_RUN_SECONDS = 40; // wall-clock seconds for a whole run at ×1
  const SIM_RHO = 1000; // kg/m³, fresh water
  const SIM_GRID = 150; // raster cells across the footprint's long axis
  const SIM_BUCKETS = 120; // distance slices used for the area ticker

  const sim = {
    open: false,
    scenario: "surge",
    model: null,
    clockS: 0,
    playing: false,
    speed: 1,
    density: 1,
    trailScale: 1,
    fx: true,
    dim: true,
    reducedMotion: false,
    lastMs: 0,
    raf: 0,
    dirty: true,
    drawnView: "",
    view: "",
    paths: null,
    spray: null,
    debris: null,
    flow: null,
    progress: 0,
    shakePx: 0,
    hudKey: "",
    hooks: false,
  };

  // Honour the OS reduced-motion setting for the cosmetic layers (shake, flow
  // dashes, breach flash). The front position and clock still advance — they
  // are the result, not decoration.
  try {
    sim.reducedMotion =
      typeof window.matchMedia === "function" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (sim.reducedMotion) sim.fx = false;
  } catch (error) {
    sim.reducedMotion = false;
  }

  function simTime(seconds) {
    const total = Math.max(0, Math.round(seconds));
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    const rest = total % 60;
    return `T+${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}`;
  }

  // Deterministic pseudo-random, so spray and radar pulses are identical on every
  // replay of a scenario instead of flickering differently each run.
  function simNoise(count, seed) {
    const values = [];
    let hash = hashString(seed);
    for (let index = 0; index < count; index += 1) {
      hash = (Math.imul(hash ^ (index + 1), 2246822519) + 374761393) >>> 0;
      values.push((hash % 100000) / 100000);
    }
    return values;
  }

  function simBuild(payload) {
    if (!mapReady()) return null;
    const dam = payload?.dam;
    const features = payload?.flood_geojson?.features || [];
    const latitude = Number(dam?.latitude);
    const longitude = Number(dam?.longitude);
    if (!features.length || !Number.isFinite(latitude) || !Number.isFinite(longitude)) return null;

    const kmPerDegLon = 111.32 * Math.max(Math.cos((latitude * Math.PI) / 180), 0.05);
    const kmPerDegLat = 110.574;
    const distanceKm = (lon, lat) =>
      Math.hypot((lon - longitude) * kmPerDegLon, (lat - latitude) * kmPerDegLat);

    // Rings in degrees, grouped by depth class, plus the bounding box the raster
    // is built over. geometryRings() is shared with the printed plan.
    const bands = BAND_COLORS.map(() => []);
    let west = Infinity;
    let south = Infinity;
    let east = -Infinity;
    let north = -Infinity;
    features.forEach((feature) => {
      const band = clamp(Number(feature?.properties?.band ?? 0), 0, BAND_COLORS.length - 1);
      geometryRings(feature.geometry).forEach((ring) => {
        const points = [];
        ring.forEach(([lon, lat]) => {
          if (!Number.isFinite(lon) || !Number.isFinite(lat)) return;
          west = Math.min(west, lon);
          east = Math.max(east, lon);
          south = Math.min(south, lat);
          north = Math.max(north, lat);
          points.push([lon, lat]);
        });
        if (points.length > 2) bands[band].push(points);
      });
    });
    if (!Number.isFinite(west) || !Number.isFinite(south)) return null;

    const allRings = bands.flat();
    let reachKm = 0;
    allRings.forEach((ring) =>
      ring.forEach(([lon, lat]) => {
        reachKm = Math.max(reachKm, distanceKm(lon, lat));
      })
    );
    if (!(reachKm > 0.5)) reachKm = 10;

    // One scanline fill of the footprint gives both the area ticker and the width
    // the HUD prints. Cells are bucketed by distance from the dam and prefix
    // summed, so "area so far" costs nothing per frame.
    const spanLon = Math.max(east - west, 1e-6);
    const spanLat = Math.max(north - south, 1e-6);
    const stepLon = spanLon / SIM_GRID;
    const stepLat = spanLat / SIM_GRID;
    const cellKm2 = stepLon * kmPerDegLon * stepLat * kmPerDegLat;
    const bucketKm = reachKm / SIM_BUCKETS;
    const bucketArea = new Array(SIM_BUCKETS).fill(0);
    let rasterKm2 = 0;

    for (let row = 0; row < SIM_GRID; row += 1) {
      const lat = south + (row + 0.5) * stepLat;
      const crossings = [];
      allRings.forEach((ring) => {
        for (let index = 0; index < ring.length; index += 1) {
          const [x1, y1] = ring[index];
          const [x2, y2] = ring[(index + 1) % ring.length];
          if ((y1 > lat) !== (y2 > lat)) {
            crossings.push(x1 + ((lat - y1) / (y2 - y1)) * (x2 - x1));
          }
        }
      });
      if (crossings.length < 2) continue;
      crossings.sort((a, b) => a - b);
      for (let index = 0; index + 1 < crossings.length; index += 2) {
        const from = crossings[index];
        const to = crossings[index + 1];
        const area = Math.max(1, Math.round((to - from) / stepLon)) * cellKm2;
        rasterKm2 += area;
        const slice = clamp(
          Math.floor(distanceKm((from + to) / 2, lat) / bucketKm),
          0,
          SIM_BUCKETS - 1
        );
        bucketArea[slice] += area;
      }
    }

    const cumulativeKm2 = [];
    let running = 0;
    bucketArea.forEach((value) => {
      running += value;
      cumulativeKm2.push(running);
    });

    const footprintKm2 = Number(payload?.summary?.estimated_flooded_area_km2) || rasterKm2;
    const uniformFallback = !(rasterKm2 > 0);

    // Surge model. The front starts at the breach celerity — peak discharge
    // through the breach section — then decays exponentially, the shape a
    // reservoir release takes as it spreads down the valley.
    const breachArea =
      Math.max(Number(payload?.inputs?.breach_head_m) || 0, 0) *
      Math.max(Number(payload?.inputs?.breach_width_m) || 0, 0);
    const peak = Number(payload?.summary?.peak_discharge_m3s) || 0;
    const v0 = clamp(breachArea > 1 && peak > 0 ? peak / breachArea : 4, 0.5, 25);
    const tauS = (reachKm * 1000) / v0;
    const endS = tauS * Math.log(1 / (1 - SIM_HORIZON));
    const frontKm = (seconds) => reachKm * (1 - Math.exp(-seconds / tauS));
    const celerity = (seconds) => v0 * Math.exp(-seconds / tauS);
    const arrivalS = (dKm) =>
      -tauS * Math.log(Math.max(1e-6, 1 - clamp(dKm / reachKm, 0, SIM_HORIZON)));

    const areaKm2 = (dKm) => {
      if (!(dKm > 0)) return 0;
      if (uniformFallback) return footprintKm2 * clamp(dKm / reachKm, 0, 1);
      return cumulativeKm2[clamp(Math.floor(dKm / bucketKm), 0, SIM_BUCKETS - 1)];
    };
    const widthKm = (dKm) =>
      bucketArea[clamp(Math.floor(dKm / bucketKm), 0, SIM_BUCKETS - 1)] / bucketKm;

    const targets = (payload?.targets || [])
      .filter(
        (target) =>
          Number.isFinite(Number(target.latitude)) && Number.isFinite(Number(target.longitude))
      )
      .map((target) => {
        const dKm = Number.isFinite(Number(target.distance_from_dam_km))
          ? Number(target.distance_from_dam_km)
          : distanceKm(Number(target.longitude), Number(target.latitude));
        return {
          label: target.label || "Area",
          lat: Number(target.latitude),
          lon: Number(target.longitude),
          dKm,
          wet: Boolean(target.inundated),
          depthM: Number(target.flood_depth_m),
          arrivalS: arrivalS(dKm),
        };
      });

    return {
      bands,
      allRings,
      dam: { lat: latitude, lon: longitude },
      targets,
      reachKm,
      footprintKm2,
      rasterKm2,
      uniformFallback,
      v0,
      tauS,
      endS,
      frontKm,
      celerity,
      arrivalS,
      areaKm2,
      widthKm,
    };
  }

  /* ---------------------------------------------------- animation rendering */

  // Rings are projected once per map view and reused every frame; re-projecting
  // several thousand vertices per frame would make the front stutter.
  function simPaths() {
    const model = sim.model;
    if (!model) return null;
    const centre = map.getCenter();
    const view = [
      map.getZoom().toFixed(2),
      centre.lat.toFixed(4),
      centre.lng.toFixed(4),
      map.getSize().x,
      map.getSize().y,
    ].join("|");
    if (sim.paths && sim.view === view) return sim.paths;

    const toPath = (rings) => {
      const path = new Path2D();
      rings.forEach((ring) => {
        ring.forEach(([lon, lat], index) => {
          const point = map.latLngToContainerPoint([lat, lon]);
          if (index === 0) path.moveTo(point.x, point.y);
          else path.lineTo(point.x, point.y);
        });
        path.closePath();
      });
      return path;
    };

    const dam = map.latLngToContainerPoint([model.dam.lat, model.dam.lon]);
    const oneKmNorth = map.latLngToContainerPoint([model.dam.lat + 1 / 110.574, model.dam.lon]);
    const pxPerKm = Math.max(Math.abs(oneKmNorth.y - dam.y), 1e-3);

    sim.paths = {
      bands: model.bands.map(toPath),
      union: toPath(model.allRings),
      dam,
      pxPerKm,
      targets: model.targets.map((target) =>
        map.latLngToContainerPoint([target.lat, target.lon])
      ),
    };
    sim.view = view;
    return sim.paths;
  }

  function simContext() {
    const canvas = $("#sim-canvas");
    if (!canvas) return null;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (!width || !height) return null;
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    // Impact shake: a dying wobble in the first seconds after the breach so the
    // release lands with weight. Deterministic sine wobble, scaled by the same
    // exponential decay as the front — the model clock is untouched. Off when
    // Impact FX is unchecked or the OS asked for reduced motion.
    if (sim.fx && !sim.reducedMotion && sim.shakePx > 0.15) {
      const now = performance.now() / 1000;
      ctx.translate(
        sim.shakePx * Math.sin(now * 31),
        sim.shakePx * Math.cos(now * 27)
      );
    }
    return ctx;
  }

  function simRoundRect(ctx, x, y, width, height, radius) {
    const r = Math.min(radius, height / 2, width / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + width - r, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + r);
    ctx.lineTo(x + width, y + height - r);
    ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
    ctx.lineTo(x + r, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  }

  function simDraw() {
    const model = sim.model;
    const ctx = simContext();
    if (!ctx || !model) return;
    const paths = simPaths();
    const { dam } = paths;
    const frontKm = model.frontKm(sim.clockS);
    const frontPx = frontKm * paths.pxPerKm;
    const scenario = sim.scenario;

    // Ghost of the modelled footprint, so the reach is legible at T+0 and after
    // the front has stopped moving.
    ctx.save();
    ctx.strokeStyle = "rgba(230, 240, 249, 0.20)";
    ctx.lineWidth = 1;
    if (scenario !== "timeline") ctx.setLineDash([5, 4]);
    ctx.stroke(paths.union);
    ctx.restore();

    if (scenario === "timeline") {
      simDrawTimeline(ctx, paths, frontPx);
      return;
    }
    if (!(frontPx > 0.6)) {
      simDrawProgress(ctx, paths);
      return;
    }

    ctx.save();
    ctx.beginPath();
    ctx.arc(dam.x, dam.y, frontPx, 0, Math.PI * 2);
    ctx.clip();

    if (scenario === "velocity") {
      // Age of the water where it lies: the front carried v0·e^(−t/τ) when it
      // passed, so the gradient runs hot at the wave front and cold behind it.
      const gradient = ctx.createRadialGradient(
        dam.x,
        dam.y,
        Math.max(frontPx * 0.03, 1),
        dam.x,
        dam.y,
        Math.max(frontPx, 1)
      );
      [
        ["#0b3f57", 0],
        ["#0ea5e9", 0.45],
        ["#22d3ee", 0.72],
        ["#fbbf24", 0.9],
        ["#f43f5e", 1],
      ].forEach(([colour, stop]) => gradient.addColorStop(stop, colour));
      ctx.globalAlpha = 0.62;
      ctx.fillStyle = gradient;
      ctx.fill(paths.union, "evenodd");
      ctx.globalAlpha = 1;
    } else {
      const alpha = scenario === "inundation" ? 0.55 : 0.42;
      for (let band = model.bands.length - 1; band >= 0; band -= 1) {
        if (!model.bands[band].length) continue;
        ctx.globalAlpha = alpha;
        ctx.fillStyle = BAND_COLORS[band];
        ctx.fill(paths.bands[band], "evenodd");
      }
      ctx.globalAlpha = 1;
    }

    // Only the inundation scenario draws a boundary; the others carry the front.
    if (scenario === "inundation") {
      ctx.strokeStyle = "rgba(34, 211, 238, 0.85)";
      ctx.lineWidth = 1.6;
      ctx.shadowColor = "rgba(34, 211, 238, 0.6)";
      ctx.shadowBlur = 10;
      ctx.stroke(paths.union);
      ctx.shadowBlur = 0;
    }
    ctx.restore();

    // The shock front itself — layered wake so it reads as water, not a ring.
    // Outer soft glow + bright core + thin foam lip just behind the edge.
    ctx.save();
    ctx.beginPath();
    ctx.arc(dam.x, dam.y, frontPx, 0, Math.PI * 2);
    ctx.strokeStyle =
      scenario === "velocity" ? "rgba(244, 63, 94, 0.28)" : "rgba(34, 211, 238, 0.28)";
    ctx.lineWidth = 9;
    ctx.stroke();
    ctx.restore();

    ctx.save();
    ctx.beginPath();
    ctx.arc(dam.x, dam.y, frontPx, 0, Math.PI * 2);
    ctx.strokeStyle =
      scenario === "velocity" ? "rgba(255, 241, 242, 0.95)" : "rgba(186, 240, 255, 0.95)";
    ctx.lineWidth = 2.2;
    ctx.shadowColor = scenario === "velocity" ? "rgba(244, 63, 94, 0.9)" : "rgba(34, 211, 238, 0.9)";
    ctx.shadowBlur = 16;
    ctx.stroke();
    ctx.restore();

    // Foam lip: a fading annulus hugging the inside of the front.
    ctx.save();
    ctx.beginPath();
    ctx.arc(dam.x, dam.y, Math.max(frontPx - 7, 0.1), 0, Math.PI * 2);
    ctx.strokeStyle =
      scenario === "velocity" ? "rgba(255, 228, 230, 0.35)" : "rgba(224, 246, 255, 0.4)";
    ctx.lineWidth = 5;
    ctx.stroke();
    ctx.restore();

    // Breach flash (first seconds) + dam beacon + drifting debris + flow
    // dashes. Purely visual: the clock, front radius and HUD stay on the model.
    if (sim.fx && !sim.reducedMotion) simDrawBreachFlash(ctx, paths);
    else simDrawDamDot(ctx, paths);
    simDrawDebris(ctx, paths, frontPx);
    if (scenario !== "timeline") simDrawFlow(ctx, paths, frontPx);

    if (scenario === "surge") simDrawSpray(ctx, paths, frontPx);
  }

  // Breach flash: a hot expanding ring in the first model-minutes that fades as
  // the surge leaves the dam. Pure screen drama — no model numbers change.
  function simDrawBreachFlash(ctx, paths) {
    const model = sim.model;
    const flashSpan = Math.max(model.endS * 0.12, 60);
    const progress = clamp(sim.clockS / flashSpan, 0, 1);
    if (progress >= 1) return;
    const fade = 1 - progress;
    ctx.save();
    ctx.beginPath();
    ctx.arc(
      paths.dam.x,
      paths.dam.y,
      6 + progress * paths.pxPerKm * 1.2,
      0,
      Math.PI * 2
    );
    ctx.fillStyle = `rgba(255, 244, 214, ${(0.28 * fade).toFixed(3)})`;
    ctx.fill();
    ctx.beginPath();
    ctx.arc(paths.dam.x, paths.dam.y, 4 + progress * paths.pxPerKm * 0.9, 0, Math.PI * 2);
    ctx.strokeStyle = `rgba(251, 191, 36, ${(0.8 * fade).toFixed(3)})`;
    ctx.lineWidth = 2.4;
    ctx.shadowColor = "rgba(251, 191, 36, 0.9)";
    ctx.shadowBlur = 18 * fade + 4;
    ctx.stroke();
    ctx.restore();

    // Dam beacon: a steady pulsing ring on the breach point.
    const beat = (sim.clockS % 2.4) / 2.4;
    ctx.save();
    ctx.beginPath();
    ctx.arc(paths.dam.x, paths.dam.y, 7, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(251, 191, 36, 0.95)";
    ctx.shadowColor = "rgba(251, 191, 36, 0.9)";
    ctx.shadowBlur = 12;
    ctx.fill();
    ctx.beginPath();
    ctx.arc(paths.dam.x, paths.dam.y, 7 + beat * 14, 0, Math.PI * 2);
    ctx.strokeStyle = `rgba(251, 191, 36, ${(0.55 * (1 - beat)).toFixed(3)})`;
    ctx.lineWidth = 1.6;
    ctx.stroke();
    ctx.restore();
  }

  // Dam marker when Impact FX is off: a calm dot instead of the flash+beacon.
  function simDrawDamDot(ctx, paths) {
    ctx.save();
    ctx.beginPath();
    ctx.arc(paths.dam.x, paths.dam.y, 5, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(251, 191, 36, 0.9)";
    ctx.strokeStyle = "rgba(9, 16, 28, 0.9)";
    ctx.lineWidth = 1.5;
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }

  // Drifting debris / foam flecks riding inside the flood, plus short flow
  // dashes streaming away from the dam. Positions come from the fixed simNoise
  // tables so replays are identical; only their progress advances with the clock.
  function simDrawDebris(ctx, paths, frontPx) {
    const model = sim.model;
    if (!sim.debris || frontPx < 4) return;
    // Particle density only thins the cosmetic flecks — never the model.
    const count = Math.floor(sim.debris.angles.length * clamp(sim.density, 0.2, 2));
    ctx.save();
    ctx.lineWidth = 1;
    for (let index = 0; index < count; index += 1) {
      const angle = sim.debris.angles[index] * Math.PI * 2;
      const lane = sim.debris.lanes[index];
      const crawl = ((sim.clockS / Math.max(model.endS, 1) + lane * 0.5) % 1 + 1) % 1;
      const radius = frontPx * (0.15 + 0.8 * crawl);
      const x = paths.dam.x + Math.cos(angle) * radius;
      const y = paths.dam.y + Math.sin(angle) * radius;
      if (!ctx.isPointInPath(paths.union, x, y, "evenodd")) continue;
      ctx.fillStyle = `rgba(230, 240, 249, ${(0.25 + sim.debris.sizes[index] * 0.45).toFixed(3)})`;
      ctx.beginPath();
      ctx.arc(x, y, 0.8 + sim.debris.sizes[index] * 1.6, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  // Flow dashes: short radial streaks streaming outward just behind the front,
  // faster and longer while the celerity is high. Clipped to the modelled
  // footprint so stray dashes never paint dry land.
  function simDrawFlow(ctx, paths, frontPx) {
    const model = sim.model;
    if (!sim.flow || frontPx < 8) return;
    const strength = clamp(model.celerity(sim.clockS) / Math.max(model.v0, 1e-6), 0, 1);
    if (strength < 0.05) return;
    // Trail length scales the dash tails; density thins how many dashes draw.
    const dashCount = Math.floor(sim.flow.angles.length * clamp(sim.density, 0.2, 2));
    ctx.save();
    ctx.beginPath();
    ctx.arc(paths.dam.x, paths.dam.y, Math.max(frontPx, 0.1), 0, Math.PI * 2);
    ctx.clip();
    for (let index = 0; index < dashCount; index += 1) {
      const angle = sim.flow.angles[index] * Math.PI * 2;
      const lane = sim.flow.lanes[index];
      const crawl = ((sim.clockS / 7 + lane) % 1 + 1) % 1;
      const radius = frontPx * (0.25 + 0.7 * crawl);
      const tail = (4 + sim.flow.lens[index] * 14) * (0.4 + strength) * sim.trailScale;
      const x = paths.dam.x + Math.cos(angle) * radius;
      const y = paths.dam.y + Math.sin(angle) * radius;
      if (!ctx.isPointInPath(paths.union, x, y, "evenodd")) continue;
      ctx.strokeStyle = `rgba(186, 240, 255, ${(0.34 * strength * (1 - crawl * 0.6)).toFixed(3)})`;
      ctx.lineWidth = 1.1 + sim.flow.lens[index];
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(
        paths.dam.x + Math.cos(angle) * Math.max(radius - tail, 0),
        paths.dam.y + Math.sin(angle) * Math.max(radius - tail, 0)
      );
      ctx.stroke();
    }
    ctx.restore();
  }

  // Spray is drawn only where the front is actually in contact with the flood,
  // using isPointInPath against the footprint, and only while the front is still
  // fast — a spent surge should not look like it is still breaking.
  // Spray count follows the particle-density setting; the celerity gate stays —
  // a spent surge never looks like it is still breaking.
  function simDrawSpray(ctx, paths, frontPx) {
    const model = sim.model;
    if (!sim.spray) return;
    const strength = clamp(model.celerity(sim.clockS) / Math.max(model.v0, 1e-6), 0, 1);
    if (strength < 0.08) return;
    const pulse = 1 - ((sim.clockS % 1.2) / 1.2);
    const sprayCount = Math.floor(sim.spray.angles.length * clamp(sim.density, 0.2, 2));
    ctx.save();
    ctx.fillStyle = `rgba(224, 246, 255, ${(0.55 * strength * pulse).toFixed(3)})`;
    for (let index = 0; index < sprayCount; index += 1) {
      const angle =
        (sim.spray.angles[index] + sim.spray.jitter[index] * 0.4) * Math.PI * 2;
      const onFrontX = paths.dam.x + Math.cos(angle) * frontPx;
      const onFrontY = paths.dam.y + Math.sin(angle) * frontPx;
      if (!ctx.isPointInPath(paths.union, onFrontX, onFrontY, "evenodd")) continue;
      const radius = frontPx * (0.95 + sim.spray.reach[index] * 0.16);
      ctx.beginPath();
      ctx.arc(
        paths.dam.x + Math.cos(angle) * radius,
        paths.dam.y + Math.sin(angle) * radius,
        0.7 + sim.spray.size[index] * 2.6 * strength,
        0,
        Math.PI * 2
      );
      ctx.fill();
    }
    ctx.restore();
  }

  function simDrawTimeline(ctx, paths, frontPx) {
    const model = sim.model;

    ctx.save();
    ctx.beginPath();
    ctx.arc(paths.dam.x, paths.dam.y, Math.max(frontPx, 0.001), 0, Math.PI * 2);
    ctx.clip();
    ctx.fillStyle = "rgba(34, 211, 238, 0.14)";
    ctx.fill(paths.union, "evenodd");
    ctx.strokeStyle = "rgba(34, 211, 238, 0.55)";
    ctx.lineWidth = 1.4;
    ctx.stroke(paths.union);
    ctx.restore();

    simDrawProgress(ctx, paths);

    // Radar pulses running out from the dam behind the front.
    for (let ring = 0; ring < 3; ring += 1) {
      const phase = ((sim.clockS / 900 + ring / 3) % 1 + 1) % 1;
      const radius = frontPx * phase;
      if (radius < 6) continue;
      ctx.save();
      ctx.beginPath();
      ctx.arc(paths.dam.x, paths.dam.y, radius, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(34, 211, 238, ${(0.45 * (1 - phase)).toFixed(3)})`;
      ctx.lineWidth = 1.3;
      ctx.stroke();
      ctx.restore();
    }

    // Pin-drop callouts, each appearing the moment the front reaches it.
    model.targets.forEach((target, index) => {
      const point = paths.targets[index];
      const arrived = sim.clockS >= target.arrivalS;
      const fresh = arrived && sim.clockS - target.arrivalS < 120;
      ctx.save();
      ctx.beginPath();
      ctx.arc(point.x, point.y, arrived ? 6 : 4, 0, Math.PI * 2);
      ctx.fillStyle = target.wet ? "rgba(244, 63, 94, 0.95)" : "rgba(9, 16, 28, 0.9)";
      ctx.strokeStyle = arrived ? "#e6f0f9" : "rgba(230, 240, 249, 0.45)";
      ctx.lineWidth = 1.6;
      if (fresh) {
        ctx.shadowColor = "#f43f5e";
        ctx.shadowBlur = 20;
      }
      ctx.fill();
      ctx.stroke();
      ctx.restore();
      if (!arrived) return;
      simDrawCallout(
        ctx,
        point,
        `${target.label} · ${simTime(target.arrivalS)} · ${
          target.wet ? `${fmt(target.depthM, 1)} m deep` : "dry"
        }`,
        target.wet
      );
    });
  }

  // Screen-space progress rail along the bottom of the map: fraction of the
  // modelled reach the front has covered. Cosmetic duplicate of the HUD
  // numbers so the eye has somewhere to sit while the surge runs.
  function simDrawProgress(ctx, paths) {
    const model = sim.model;
    if (!model || !ctx.canvas) return;
    // Canvas coordinates are CSS pixels (the dpr scale is in the transform),
    // so clientWidth/Height map 1:1 onto drawing units.
    const width = ctx.canvas.clientWidth;
    const height = ctx.canvas.clientHeight;
    const barW = Math.min(300, Math.max(140, width * 0.42));
    const barH = 3;
    const x = (width - barW) / 2;
    const y = height - 108;
    if (y < 20) return;
    ctx.save();
    ctx.fillStyle = "rgba(9, 16, 28, 0.75)";
    simRoundRect(ctx, x - 8, y - 18, barW + 16, 34, 8);
    ctx.fill();
    ctx.fillStyle = "rgba(230, 240, 249, 0.5)";
    ctx.font = '600 10px "JetBrains Mono", ui-monospace, Consolas, monospace';
    ctx.fillText(`RUN ${fmt(sim.progress * 100, 0)} %`, x, y - 6);
    ctx.fillStyle = "rgba(148, 190, 232, 0.25)";
    simRoundRect(ctx, x, y, barW, barH, barH / 2);
    ctx.fill();
    const fill = barW * clamp(sim.progress, 0, 1);
    if (fill > 0.5) {
      const gradient = ctx.createLinearGradient(x, 0, x + barW, 0);
      gradient.addColorStop(0, "rgba(34, 211, 238, 0.9)");
      gradient.addColorStop(1, "rgba(251, 191, 36, 0.95)");
      ctx.fillStyle = gradient;
      simRoundRect(ctx, x, y, fill, barH, barH / 2);
      ctx.fill();
    }
    ctx.restore();
  }

  function simDrawCallout(ctx, point, text, alert) {
    ctx.save();
    ctx.font = '600 11px "JetBrains Mono", ui-monospace, Consolas, monospace';
    const padding = 7;
    const width = ctx.measureText(text).width + padding * 2;
    const x = point.x + 13;
    const y = point.y - 30;
    const leader = alert ? "rgba(244, 63, 94, 0.9)" : "rgba(148, 190, 232, 0.5)";
    ctx.strokeStyle = leader;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(point.x + 4, point.y - 5);
    ctx.lineTo(x + 5, y + 19);
    ctx.stroke();
    ctx.fillStyle = "rgba(9, 16, 28, 0.92)";
    ctx.strokeStyle = leader;
    simRoundRect(ctx, x, y, width, 21, 5);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = alert ? "#ffe4e6" : "#e6f0f9";
    ctx.fillText(text, x + padding, y + 14.5);
    ctx.restore();
  }

  /* --------------------------------------------------------- animation HUD */

  function simRenderHud() {
    const model = sim.model;
    const hud = $("#sim-hud");
    if (!model || !hud) return;
    const scenario = SIM_SCENARIOS.find((entry) => entry.id === sim.scenario) || SIM_SCENARIOS[0];
    const frontKm = model.frontKm(sim.clockS);
    const speed = model.celerity(sim.clockS);
    const forceKn = (0.5 * SIM_RHO * speed * speed) / 1000; // kN/m² (kPa)
    const area = model.areaKm2(frontKm);
    const share = model.footprintKm2 > 0 ? (area / model.footprintKm2) * 100 : 0;
    const reached = model.targets.filter((target) => sim.clockS >= target.arrivalS).length;

    const rows = [["Front distance", `${fmt(frontKm, 2)} km of ${fmt(model.reachKm, 1)} km`]];
    if (sim.scenario === "velocity") {
      rows.push(
        ["Flow speed at the front", `${fmt(speed, 2)} m/s`],
        ["Impact force ½ρv²", `${fmt(forceKn, 1)} kN/m²`],
        ["Flooded width here", `${fmt(model.widthKm(frontKm), 2)} km`]
      );
    } else if (sim.scenario === "inundation") {
      rows.push(
        ["Total area inundated", `${fmt(area, 2)} km²`],
        ["Share of the footprint", `${fmt(share, 0)} %`]
      );
    } else if (sim.scenario === "surge") {
      rows.push(["Front celerity now", `${fmt(speed, 2)} m/s`]);
    } else {
      rows.push(["Areas reached", `${reached} of ${model.targets.length}`]);
    }

    const arrivals = model.targets
      .slice()
      .sort((a, b) => a.arrivalS - b.arrivalS)
      .map((target) => {
        const there = sim.clockS >= target.arrivalS;
        const status = there
          ? target.wet
            ? `reached ${simTime(target.arrivalS)} · ${fmt(target.depthM, 1)} m deep`
            : `dry when the front passed, ${simTime(target.arrivalS)}`
          : `eta ${simTime(target.arrivalS)}`;
        return `<li class="${there ? "is-reached" : ""}"><span>${esc(target.label)}</span><em>${esc(status)}</em></li>`;
      })
      .join("");

    // Repainting this panel every frame would be wasted work: only numbers that
    // actually moved between frames justify a rebuild.
    const key = [sim.scenario, sim.clockS.toFixed(0), frontKm.toFixed(2), reached].join("|");
    if (key === sim.hudKey) return;
    sim.hudKey = key;

    const source = model.uniformFallback
      ? `Front celerity ${fmt(model.v0, 2)} m/s at the breach, decaying over ` +
        `${fmt(model.tauS / 60, 0)} min. The footprint could not be rasterised, so the ` +
        `area ticker is scaled uniformly along the reach.`
      : `Front celerity ${fmt(model.v0, 2)} m/s at the breach (peak discharge ÷ breach ` +
        `section), decaying over ${fmt(model.tauS / 60, 0)} min; reach ` +
        `${fmt(model.reachKm, 1)} km. Areas summed from a ${SIM_GRID}-cell raster of the ` +
        `exported polygons — ${fmt(model.rasterKm2, 0)} km² against the model's ` +
        `${fmt(model.footprintKm2, 0)} km².`;

    hud.innerHTML = `
      <h4 class="sim__title">${esc(scenario.title)}</h4>
      <p class="sim__blurb">${esc(scenario.blurb)}</p>
      <dl class="sim__rows">
        ${rows
          .map(([label, value]) => `<div><dt>${esc(label)}</dt><dd>${esc(value)}</dd></div>`)
          .join("")}
      </dl>
      ${arrivals ? `<ul class="sim__arrivals">${arrivals}</ul>` : ""}
      <p class="sim__source">${esc(source)}</p>`;
  }

  function simUpdateClock() {
    const clock = $("#sim-clock");
    if (clock) clock.textContent = simTime(sim.clockS);
  }

  function simSetPlaying(playing) {
    sim.playing = playing;
    sim.lastMs = 0;
    if (playing && sim.model && sim.clockS >= sim.model.endS) sim.clockS = 0;
    const button = $("#sim-play");
    if (button) {
      button.textContent = playing ? "Pause" : "Play";
      button.setAttribute("aria-pressed", String(playing));
    }
    sim.dirty = true;
  }

  function simSetScenario(id) {
    if (!SIM_SCENARIOS.some((entry) => entry.id === id)) return;
    sim.scenario = id;
    sim.dirty = true;
    $$("#sim-scenarios [data-sim-scenario]").forEach((button) =>
      button.classList.toggle("is-active", button.dataset.simScenario === id)
    );
  }

  /* ---------------------------------------------------- animation transport */

  function simFrame(nowMs) {
    if (!sim.open || !sim.model) {
      sim.raf = 0;
      return;
    }
    if (sim.playing) {
      const dt = sim.lastMs ? Math.min((nowMs - sim.lastMs) / 1000, 0.3) : 0;
      // One wall-clock second carries endS / SIM_RUN_SECONDS of model time, so a
      // whole breach replays in about forty seconds at ×1 whatever the reach is.
      // The HUD prints celerity and clock, never a claim of real time.
      sim.clockS += dt * (sim.model.endS / SIM_RUN_SECONDS) * sim.speed;
      if (sim.clockS >= sim.model.endS) {
        sim.clockS = sim.model.endS;
        simSetPlaying(false);
      }
      // Cosmetic meters: progress fraction plus a dying shake in the first
      // model-minutes. Force repaint every tick while the clock runs (or the
      // shake is still alive) so the motion stays smooth.
      sim.progress = clamp(
        sim.model.frontKm(sim.clockS) / Math.max(sim.model.reachKm, 1e-6),
        0,
        1
      );
      sim.shakePx = 5 * Math.exp(-sim.clockS / Math.max(sim.model.tauS * 0.25, 30));
      sim.dirty = true;
    } else if (sim.shakePx > 0.15) {
      sim.shakePx *= 0.9;
      sim.dirty = true;
    }
    sim.lastMs = nowMs;
    if (sim.dirty || sim.view !== sim.drawnView) {
      simDraw();
      simRenderHud();
      simUpdateClock();
      sim.drawnView = sim.view;
      sim.dirty = false;
    }
    sim.raf = requestAnimationFrame(simFrame);
  }

  function simHooks() {
    if (sim.hooks || !mapReady()) return;
    sim.hooks = true;
    map.on("move zoom viewreset resize", () => {
      sim.dirty = true;
    });
  }

  function simApplyDim() {
    if (!floodLayer) return;
    floodLayer.setStyle(
      sim.open && sim.dim
        ? { opacity: 0.18, fillOpacity: 0.06 }
        : { opacity: 1, fillOpacity: 0.35 }
    );
    sim.dirty = true;
  }

  function simOpen(scenario) {
    if (!state.payload) {
      toast("Run a screening first — the animation replays its result.", "warn");
      return;
    }
    const model = simBuild(state.payload);
    if (!model) {
      toast("There is no flood footprint to animate yet.", "warn");
      return;
    }
    sim.model = model;
    sim.spray = {
      angles: simNoise(90, "spray-angle"),
      jitter: simNoise(90, "spray-jitter"),
      reach: simNoise(90, "spray-reach"),
      size: simNoise(90, "spray-size"),
    };
    sim.debris = {
      angles: simNoise(70, "debris-angle"),
      lanes: simNoise(70, "debris-lane"),
      sizes: simNoise(70, "debris-size"),
    };
    sim.flow = {
      angles: simNoise(60, "flow-angle"),
      lanes: simNoise(60, "flow-lane"),
      lens: simNoise(60, "flow-len"),
    };
    sim.paths = null;
    sim.view = "";
    sim.drawnView = "";
    sim.hudKey = "";
    sim.clockS = 0;
    sim.progress = 0;
    sim.shakePx = 5;
    sim.open = true;
    $("#sim").hidden = false;
    // The HUD takes the legend's corner, and Leaflet's own fill is dimmed so the
    // canvas layer is what the eye follows (unless Dim map is unchecked).
    $("#legend").hidden = true;
    simApplyDim();
    simSetScenario(scenario || sim.scenario);
    simSetPlaying(true);
    simHooks();
    if (!sim.raf) sim.raf = requestAnimationFrame(simFrame);
  }

  function simClose() {
    if (!sim.open) return;
    sim.open = false;
    sim.playing = false;
    if (sim.raf) cancelAnimationFrame(sim.raf);
    sim.raf = 0;
    $("#sim").hidden = true;
    simApplyDim();
    if (state.payload) $("#legend").hidden = false;
    sim.model = null;
    sim.paths = null;
  }

  function simInit() {
    const list = $("#sim-scenarios");
    if (list) {
      list.innerHTML = SIM_SCENARIOS.map(
        (entry) =>
          `<button type="button" class="segmented__btn" data-sim-scenario="${entry.id}">` +
          `${esc(entry.label)}</button>`
      ).join("");
    }
    simSetScenario("surge");
    const density = $("#sim-density");
    if (density) {
      sim.density = Number(density.value) || 1;
      density.addEventListener("change", (event) => {
        sim.density = clamp(Number(event.target.value) || 1, 0.2, 2);
        sim.dirty = true;
      });
    }
    const trails = $("#sim-trails");
    if (trails) {
      sim.trailScale = Number(trails.value) || 1;
      trails.addEventListener("change", (event) => {
        sim.trailScale = clamp(Number(event.target.value) || 1, 0.2, 2.5);
        sim.dirty = true;
      });
    }
    const fx = $("#sim-fx");
    if (fx) {
      if (sim.reducedMotion) fx.checked = false;
      sim.fx = fx.checked && !sim.reducedMotion;
      fx.addEventListener("change", (event) => {
        sim.fx = event.target.checked && !sim.reducedMotion;
        sim.dirty = true;
      });
    }
    const dim = $("#sim-dim");
    if (dim) {
      sim.dim = dim.checked;
      dim.addEventListener("change", (event) => {
        sim.dim = event.target.checked;
        simApplyDim();
      });
    }
  }

  /* -------------------------------------------------------------- workflow */
  function setBusy(busy) {
    state.busy = busy;
    const button = $("#run-button");
    button.disabled = busy;
    button.classList.toggle("is-loading", busy);
    $(".btn__text", button).textContent = busy ? "Simulating…" : "Run screening";
  }

  async function runSimulation(event) {
    event.preventDefault();
    if (state.busy) return;

    const errorNote = $("#error-note");
    errorNote.hidden = true;

    const inputs = collectInputs();
    const problem = validate(inputs);
    if (problem) {
      errorNote.hidden = false;
      errorNote.textContent = problem;
      toast(problem, "error");
      return;
    }

    setBusy(true);
    try {
      let payload;
      if (state.mode === "live") {
        payload =
          state.terrainSource === "upload"
            ? await runUpload(inputs)
            : await runAuto(inputs);
      } else {
        await new Promise((resolve) => setTimeout(resolve, 650));
        payload = demoSimulate(inputs);
      }

      state.payload = payload;
      renderResults(payload);
      renderSummaryChips(payload);
      renderTargetStatuses(payload.targets);
      renderPrintHead(payload);
      drawFlood(payload);
      flagDamSnap(payload);
      $("#btn-export").disabled = false;
      $("#btn-copy").disabled = false;
      $("#btn-print").disabled = false;
      $("#btn-anim").disabled = false;
      // A new result invalidates the replay, so rebuild it from the new payload.
      if (sim.open) simOpen(sim.scenario);

      const affected = payload.targets.filter((target) => target.inundated).length;
      const exposureNote =
        payload.exposure?.available && payload.exposure.assets_inundated
          ? ` · ${payload.exposure.assets_inundated} assets affected`
          : "";
      toast(
        `${payload.dam.name}: ${fmt(payload.summary.estimated_flooded_area_km2, 1)} km² flooded · ` +
          `${affected}/${payload.targets.length} areas${exposureNote}`,
        "success"
      );
    } catch (error) {
      errorNote.hidden = false;
      errorNote.textContent = error.message;
      toast(error.message, "error", 7000);
    } finally {
      setBusy(false);
    }
  }

  function exportGeoJSON() {
    if (!state.payload) return;
    const payload = state.payload;

    const collection = {
      type: "FeatureCollection",
      properties: {
        generator: "SIH26161 Flood Screening",
        exported_at: new Date().toISOString(),
        dam: payload.dam,
        inputs: payload.inputs,
        summary: payload.summary,
        terrain: payload.terrain,
        notes: payload.notes,
        disclaimer: payload.warning,
      },
      features: [
        ...payload.flood_geojson.features,
        ...payload.targets
          .filter((target) => target.inside_grid)
          .map((target) => ({
            type: "Feature",
            properties: { kind: "target", ...target },
            geometry: { type: "Point", coordinates: [target.longitude, target.latitude] },
          })),
        ...(payload.exposure?.available
          ? payload.exposure.assets
              .filter((asset) => asset.inundated)
              .map((asset) => ({
                type: "Feature",
                properties: { kind: "exposed_asset", ...asset },
                geometry: { type: "Point", coordinates: [asset.longitude, asset.latitude] },
              }))
          : []),
      ],
    };

    const blob = new Blob([JSON.stringify(collection, null, 2)], {
      type: "application/geo+json",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${payload.dam.dam_id}_${Math.round(payload.inputs.release_level_m || 0)}m_flood.geojson`;
    link.click();
    URL.revokeObjectURL(url);
    toast("GeoJSON export downloaded.", "success");
  }

  /* -------------------------------------------------- results placeholder
     The panel is never left showing a blank "nothing loaded" card: the
     default scenario runs automatically on load, and a reset reloads it. */

  const PLACEHOLDER_TEXT =
    "Loading the default scenario — fetching terrain and solving the flood " +
    "footprint.";

  function showPlaceholder(message) {
    const box = $("#results-empty");
    if (!box) return;
    const text = $("#results-empty-text");
    if (text) text.textContent = message || PLACEHOLDER_TEXT;
    box.hidden = false;
  }

  function hidePlaceholder() {
    const box = $("#results-empty");
    if (box) box.hidden = true;
  }

  let autoRunQueued = false;

  function autoRunDefault() {
    if (autoRunQueued) return;
    autoRunQueued = true;
    window.setTimeout(() => {
      autoRunQueued = false;
      if (state.busy) return;
      const form = $("#simulation-form");
      if (!form) return;
      if (typeof form.requestSubmit === "function") form.requestSubmit();
      else form.dispatchEvent(new Event("submit", { cancelable: true, bubbles: true }));
    }, 140);
  }

  function resetScenario() {
    state.payload = null;
    state.demFile = null;
    state.widthEdited = false;
    state.headEdited = false;
    state.levelEdited = false;
    state.volumeEdited = false;
    $("#simulation-form").reset();
    $("#file-chip").hidden = true;
    $("#dem").value = "";
    $("#results-body").hidden = true;
    showPlaceholder(PLACEHOLDER_TEXT);
    $("#summary-chips").hidden = true;
    $("#legend").hidden = true;
    $("#btn-export").disabled = true;
    $("#btn-copy").disabled = true;
    $("#btn-print").disabled = true;
    $("#btn-anim").disabled = true;
    simClose();
    $$("#print-head, #print-scenario, #print-map, #print-foot").forEach(
      (element) => (element.innerHTML = "")
    );
    $("#error-note").hidden = true;
    $("#snap-note").hidden = true;
    $("#upload-block").hidden = true;

    if (floodLayer && mapReady()) {
      map.removeLayer(floodLayer);
      floodLayer = null;
    }
    if (assetLayer && mapReady()) {
      map.removeLayer(assetLayer);
      assetLayer = null;
    }

    [1, 2].forEach((index) => {
      if (targetMarkers[index] && mapReady()) {
        map.removeLayer(targetMarkers[index]);
        targetMarkers[index] = null;
      }
      const fieldset = $(`.target[data-target="${index}"]`);
      fieldset.classList.remove("is-flooded", "is-dry");
      $(`[data-status="${index}"]`).textContent = "Not simulated yet";
    });

    $$('input[type="range"]').forEach(paintRange);
    $("#breach-output").textContent = `${$("#breach-head").value} m`;
    $("#attenuation-output").textContent = `${Number($("#attenuation").value).toFixed(2)} m/km`;
    $("#radius-output").textContent = `${$("#radius").value} km`;

    populateDamSelect(state.dams);
    setPickMode(null);
    // Reload the default scenario so the panel never sits empty.
    autoRunDefault();
  }

  /* ----------------------------------------------------------- event wiring */
  function wireEvents() {
    $("#simulation-form").addEventListener("submit", runSimulation);
    $("#btn-reset").addEventListener("click", resetScenario);
    $("#btn-export").addEventListener("click", exportGeoJSON);
    $("#btn-copy").addEventListener("click", copySummary);
    $("#btn-print").addEventListener("click", printReport);

    // Animation transport. Esc closes it, and the space bar toggles playback
    // when nothing else on the page has focus.
    $("#btn-anim").addEventListener("click", () => {
      if (sim.open) simClose();
      else simOpen(sim.scenario);
    });
    $("#sim-close").addEventListener("click", simClose);
    $("#sim-restart").addEventListener("click", () => {
      sim.clockS = 0;
      sim.hudKey = "";
      simSetPlaying(true);
    });
    $("#sim-play").addEventListener("click", () => simSetPlaying(!sim.playing));
    $("#sim-speed").addEventListener("change", (event) => {
      sim.speed = Number(event.target.value) || 1;
    });
    $("#sim-scenarios").addEventListener("click", (event) => {
      const button = event.target.closest("[data-sim-scenario]");
      if (button) simSetScenario(button.dataset.simScenario);
    });
    window.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && sim.open) simClose();
      if (event.code === "Space" && sim.open && event.target === document.body) {
        event.preventDefault();
        simSetPlaying(!sim.playing);
      }
    });

    // Fires for Ctrl+P and window.print() alike, so the masthead is always
    // current when the print dialog paints.
    window.addEventListener("beforeprint", () => {
      if (state.payload) renderPrintHead(state.payload);
    });

    $("#dam").addEventListener("change", applySelectedDam);

    $("#release-level").addEventListener("input", () => {
      state.levelEdited = true;
    });
    $("#release-volume").addEventListener("input", () => {
      state.volumeEdited = true;
    });

    const radius = $("#radius");
    radius.addEventListener("input", () => {
      $("#radius-output").textContent = `${radius.value} km`;
      paintRange(radius);
    });

    const head = $("#breach-head");
    head.addEventListener("input", () => {
      state.headEdited = true;
      $("#breach-output").textContent = `${head.value} m`;
      paintRange(head);
      if (!state.widthEdited) $("#breach-width").value = head.value;
    });

    $("#breach-width").addEventListener("input", () => {
      state.widthEdited = true;
    });

    const attenuation = $("#attenuation");
    attenuation.addEventListener("input", () => {
      $("#attenuation-output").textContent = `${Number(attenuation.value).toFixed(2)} m/km`;
      paintRange(attenuation);
    });

    $("#attenuation-mode").addEventListener("change", (event) => {
      $("#attenuation-field").hidden = event.target.value !== "manual";
    });

    $$('input[name="terrain_source"]').forEach((radio) =>
      radio.addEventListener("change", (event) => {
        state.terrainSource = event.target.value;
        $("#upload-block").hidden = state.terrainSource !== "upload";
      })
    );

    [1, 2].forEach((index) => {
      [`#area${index}-lat`, `#area${index}-lon`].forEach((selector) => {
        $(selector).addEventListener("change", () => updateTargetMarker(index));
      });
    });

    $$(".btn--pick").forEach((button) =>
      button.addEventListener("click", () => {
        const index = Number(button.dataset.pick);
        setPickMode(state.pickTarget === index ? null : index);
      })
    );

    const dropzone = $("#dropzone");
    const demInput = $("#dem");

    const acceptFile = (file) => {
      if (!file) return;
      if (!/\.(tif|tiff|geotiff)$/i.test(file.name)) {
        toast("Only GeoTIFF files (.tif/.tiff) are accepted.", "error");
        return;
      }
      state.demFile = file;
      $("#file-name").textContent = file.name;
      $("#file-size").textContent = `${(file.size / 1024 / 1024).toFixed(1)} MB`;
      $("#file-chip").hidden = false;
      toast(`${file.name} ready to upload.`, "success");
    };

    dropzone.addEventListener("click", () => demInput.click());
    dropzone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        demInput.click();
      }
    });
    demInput.addEventListener("change", () => acceptFile(demInput.files[0]));

    ["dragenter", "dragover"].forEach((type) =>
      dropzone.addEventListener(type, (event) => {
        event.preventDefault();
        dropzone.classList.add("is-dragging");
      })
    );
    ["dragleave", "drop"].forEach((type) =>
      dropzone.addEventListener(type, (event) => {
        event.preventDefault();
        dropzone.classList.remove("is-dragging");
      })
    );
    dropzone.addEventListener("drop", (event) => acceptFile(event.dataTransfer.files[0]));

    $("#btn-clear-file").addEventListener("click", () => {
      state.demFile = null;
      demInput.value = "";
      $("#file-chip").hidden = true;
    });

    $$(".segmented__btn").forEach((button) =>
      button.addEventListener("click", () => setBasemap(button.dataset.basemap))
    );

    const dialog = $("#settings-dialog");
    $("#btn-settings").addEventListener("click", () => {
      $("#api-url").value = state.apiUrl;
      $("#api-test-status").textContent = "";
      dialog.showModal();
    });
    $("#api-status").addEventListener("click", () => $("#btn-settings").click());

    $("#btn-test-api").addEventListener("click", async () => {
      const url = $("#api-url").value.trim() || FALLBACK.apiUrl;
      const status = $("#api-test-status");
      status.textContent = "Testing…";
      state.apiUrl = url;
      try {
        await detectMode({ silent: true });
        if (state.mode !== "live") throw new Error("unreachable");
        localStorage.setItem("flood.apiUrl", url);
        state.dams = await loadDams();
        populateDamSelect(state.dams);
        status.textContent = "✓ connected";
        toast("Connected to the screening API.", "success");
      } catch {
        status.textContent = "✗ unreachable";
      }
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && state.pickTarget) setPickMode(null);
    });
  }

  /* ------------------------------------------------------------------ init */
  async function init() {
    initMap();
    wireEvents();
    simInit();
    renderLegend();
    $$('input[type="range"]').forEach(paintRange);
    $("#breach-output").textContent = `${$("#breach-head").value} m`;
    $("#radius-output").textContent = `${$("#radius").value} km`;
    $("#attenuation-output").textContent = `${Number($("#attenuation").value).toFixed(2)} m/km`;

    // Silent: the status pill and the mode banner already explain demo mode, so
    // a toast on every load is noise.
    await detectMode({ silent: true });

    try {
      state.dams = await loadDams();
      populateDamSelect(state.dams);
      updateTargetMarker(1);
      updateTargetMarker(2);
    } catch (error) {
      $("#dam-meta").textContent = error.message;
      toast(error.message, "error");
    }

    // Open on a solved scenario rather than an empty panel.
    showPlaceholder(PLACEHOLDER_TEXT);
    autoRunDefault();
  }

  // app.js is the last element in <body>, so the DOM is already parsed.
  init();
})();