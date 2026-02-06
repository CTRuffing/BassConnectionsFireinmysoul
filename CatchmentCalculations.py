#!/usr/bin/env python3
"""
catchment_calculations_rewrite.py

Rewritten / hardened version of your catchment-area script.
Features:
- Robust hospital name matching (handles Al Quds and Kuwait)
- Includes Al Quds and Kuwait in calculations and visualization
- Voronoi with shapely.voronoi_diagram when available; SciPy fallback otherwise
- Two-week period aggregation with weighted areas if hospital status changes mid-period
- Outputs:
    1) Excel: catchment_areas_over_time.xlsx
    2) HTML: first period visualization saved to catchment_visualization_YYYYMMDD_to_YYYYMMDD.html
Requirements: geopandas, shapely, pandas, numpy, pyproj, scipy, folium, openpyxl
"""

from datetime import datetime, timedelta
import os
import re
import warnings
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon, box
from shapely.ops import unary_union
from shapely import geometry as shapely_geom
from pyproj import Geod
from scipy.spatial import Voronoi
import folium

warnings.filterwarnings("ignore")

# Try shapely voronoi
try:
    from shapely.ops import voronoi_diagram  # type: ignore
    _HAS_SHAPELY_VORONOI = True
except Exception:
    voronoi_diagram = None
    _HAS_SHAPELY_VORONOI = False

# -------------------------
# Config
# -------------------------
HOSP_PATH = "Hospitals_OpenCloseoverTime.xlsx"
GAZA_BOUNDARY = "gaza_boundary.geojson"
OUTPUT_DIR = "."
CRS_WGS84 = "EPSG:4326"
CRS_WEBMERC = "EPSG:3857"
CATCHMENT_DISTANCE_KM = 5.0
VORONOI_BUFFER_METERS = 15000  # generous buffer around Gaza bbox
# canonical targets and mapping
HOSPITAL_NAME_MAPPING = {
    "Al Nasser": ["Al Nasser", "Nasser Hospital", "Nasser"],
    "EGH": ["EGH", "European Hospital", "European", "El European"],
    "Al Shifa": ["Al Shifa", "Al Shifa Medical Hospital", "Shifa"],
    "Al Quds": ["Al Quds", "Al-Quds", "Al Quds Hospital", "Al-Quds Hospital"],
    "Kuwait": ["Kuwait", "Kuwait Hospital", "Kuwait Hosp"]
}
TARGET_HOSPITALS = list(HOSPITAL_NAME_MAPPING.keys())

# Colors for visualization (extend/adjust as desired)
HOSPITAL_COLORS = {
    "Al Nasser": "#e41a1c",
    "EGH": "#377eb8",
    "Al Shifa": "#4daf4a",
    "Al Quds": "#984ea3",
    "Kuwait": "#ff7f00"
}


# -------------------------
# Helpers
# -------------------------
def _normalize(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s)
    # replace non-alphanumeric with spaces, collapse multiple spaces, lower
    s2 = re.sub(r"[^0-9a-zA-Z]+", " ", s).strip().lower()
    s2 = re.sub(r"\s+", " ", s2)
    return s2


def match_hospital_name(name: Optional[str]) -> Optional[str]:
    """
    Robust matching to canonical hospital names in HOSPITAL_NAME_MAPPING.
    Returns canonical name (e.g., "Al Quds") or None.
    """
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return None
    name_norm = _normalize(name)
    if not name_norm:
        return None

    # Exact normalized match against variants
    for canonical, variants in HOSPITAL_NAME_MAPPING.items():
        for v in variants + [canonical]:
            if _normalize(v) == name_norm:
                return canonical

    # Token overlap heuristic: require at least one significant token match
    name_tokens = set(name_norm.split())
    stop_tokens = {"al", "the", "hospital", "hosp", "medical"}
    for canonical, variants in HOSPITAL_NAME_MAPPING.items():
        tokens = set()
        tokens.update(_normalize(canonical).split())
        for v in variants:
            tokens.update(_normalize(v).split())
        sig_overlap = [t for t in (tokens & name_tokens) if t not in stop_tokens]
        if sig_overlap:
            return canonical

    # final simple substring heuristics
    if "nasser" in name_norm:
        return "Al Nasser"
    if "european" in name_norm or "egh" in name_norm:
        return "EGH"
    if "shifa" in name_norm:
        return "Al Shifa"
    if "quds" in name_norm or "al quds" in name_norm:
        return "Al Quds"
    if "kuwait" in name_norm:
        return "Kuwait"

    return None


# -------------------------
# Input parsing
# -------------------------
def read_hospitals_table(path: str) -> Tuple[pd.DataFrame, List[Tuple[str, str, Optional[datetime]]]]:
    """
    Parse the Hospitals_OpenCloseoverTime.xlsx file.
    Returns: hospitals_df (with 'Hospital','lon','lat' and schedule cols), schedule_meta list of tuples (col, type, date)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find hospitals file: {path}")

    # Read two-row header preview (to try parse dates) and full table
    raw_head = pd.read_excel(path, header=None, nrows=2)
    full = pd.read_excel(path, header=0)

    cols = list(full.columns)

    def find_col(opts: List[str]) -> Optional[str]:
        for o in opts:
            for c in cols:
                if isinstance(c, str) and c.strip().lower().startswith(o.lower()):
                    return c
        return None

    hosp_col = find_col(["hospital", "name"])
    lon_col = find_col(["longitude (x)", "longitude", "lon", "x"])
    lat_col = find_col(["latitude (y)", "latitude", "lat", "y"])
    if hosp_col is None or lon_col is None or lat_col is None:
        raise ValueError(f"Couldn't detect Hospital/lon/lat columns. Found: {cols}")

    # Detect Open/Closed schedule columns using header names + date in second row
    schedule_meta: List[Tuple[str, str, Optional[datetime]]] = []
    for c in cols:
        if isinstance(c, str):
            lc = c.strip().lower()
            if lc.startswith("open") or lc.startswith("closed"):
                typ = "Open" if lc.startswith("open") else "Closed"
                dt = None
                try:
                    idx = cols.index(c)
                    raw = raw_head.iat[1, idx]
                    if pd.notnull(raw):
                        dt = pd.to_datetime(raw)
                except Exception:
                    dt = None
                schedule_meta.append((c, typ, dt))

    # keep only schedule columns with detected date
    schedule_meta = [(c, typ, d) for (c, typ, d) in schedule_meta if d is not None]

    hospitals_df = full[[hosp_col, lon_col, lat_col]].rename(columns={hosp_col: "Hospital", lon_col: "lon", lat_col: "lat"})
    hospitals_df["lon"] = pd.to_numeric(hospitals_df["lon"], errors="coerce")
    hospitals_df["lat"] = pd.to_numeric(hospitals_df["lat"], errors="coerce")

    # copy schedule marker columns into hospitals_df (if present)
    for c, _, _ in schedule_meta:
        if c in full.columns:
            hospitals_df[c] = full[c]

    return hospitals_df, schedule_meta


def build_hospital_open_intervals(hospitals_df: pd.DataFrame, schedule_meta: List[Tuple[str, str, Optional[datetime]]]) -> Dict[str, List[Tuple[datetime, str]]]:
    """
    Build per-hospital ordered list of (date, status) changes.
    If no per-row marks are present, global schedule_meta events are used for all hospitals.
    """
    events = sorted([(pd.to_datetime(d).to_pydatetime(), col, typ) for (col, typ, d) in schedule_meta], key=lambda x: x[0])
    hospital_intervals: Dict[str, List[Tuple[datetime, str]]] = {}

    for _, row in hospitals_df.iterrows():
        name = row["Hospital"]
        changes: List[Tuple[datetime, str]] = []

        # per-row markers
        for dt, col, typ in events:
            col_name = col
            val = row.get(col_name, None) if col_name in row.index else None
            if pd.notnull(val) and str(val).strip() != "":
                changes.append((dt, typ))

        # if no per-row markers, fall back to global events
        if not changes:
            for dt, col, typ in events:
                changes.append((dt, typ))

        # compress duplicates and sort
        changes_sorted = sorted(changes, key=lambda x: x[0])
        compressed: List[Tuple[datetime, str]] = []
        for dt, typ in changes_sorted:
            if not compressed or compressed[-1][1] != typ:
                compressed.append((dt, typ))

        if not compressed:
            compressed = [(datetime(1900, 1, 1), "Open")]

        hospital_intervals[name] = compressed

    return hospital_intervals


def get_hospital_status_at_date(hospital_intervals: Dict[str, List[Tuple[datetime, str]]], hospital_name: str, date: datetime) -> str:
    """
    Return "Open" or "Closed" for hospital_name at the given date.
    """
    if hospital_name not in hospital_intervals:
        return "Closed"
    changes = hospital_intervals[hospital_name]
    last_status = "Open"
    for dt, typ in changes:
        if dt <= date:
            last_status = typ
        else:
            break
    return last_status


# -------------------------
# Voronoi / Catchment functions
# -------------------------
def _scipy_voronoi_fallback(pts_wgs: gpd.GeoDataFrame, pts_proj: gpd.GeoDataFrame, clip_union, bbox_proj) -> gpd.GeoDataFrame:
    """Fallback Voronoi using SciPy in projected coords (WEBMERC)."""
    coords = np.array([(pt.x, pt.y) for pt in pts_proj.geometry])
    vor = Voronoi(coords)

    proj_polys = []
    for pt_idx, region_index in enumerate(vor.point_region):
        region = vor.regions[region_index]
        if not region or -1 in region:
            # unbounded region: create convex hull from site + bbox coords
            site = coords[pt_idx]
            bbox_coords = np.array(bbox_proj.exterior.coords)
            pts_for_hull = np.vstack([site, bbox_coords])
            hull = shapely_geom.MultiPoint([tuple(p) for p in pts_for_hull]).convex_hull
            poly = hull
        else:
            try:
                verts = [vor.vertices[i] for i in region]
                poly = Polygon(verts)
            except Exception:
                site = coords[pt_idx]
                bbox_coords = np.array(bbox_proj.exterior.coords)
                pts_for_hull = np.vstack([site, bbox_coords])
                hull = shapely_geom.MultiPoint([tuple(p) for p in pts_for_hull]).convex_hull
                poly = hull
        poly_clipped = poly.intersection(bbox_proj)
        proj_polys.append(poly_clipped)

    polys_proj_gdf = gpd.GeoDataFrame(geometry=proj_polys, crs=CRS_WEBMERC)
    polys_wgs = polys_proj_gdf.to_crs(CRS_WGS84).reset_index(drop=True)

    hosp_names = pts_wgs['Hospital'].values
    rows = []
    for i, poly in enumerate(polys_wgs.geometry):
        try:
            clipped = poly.intersection(clip_union)
        except Exception:
            clipped = Polygon()
        rows.append({'Hospital': hosp_names[i], 'geometry': (clipped if clipped is not None else Polygon())})

    result = gpd.GeoDataFrame(rows, crs=CRS_WGS84)

    # assign any leftover pieces to nearest hospital
    covered = unary_union([g for g in result.geometry if g is not None and not g.is_empty]) if not result.empty else None
    leftover = clip_union.difference(covered) if covered is not None else clip_union
    if leftover and not leftover.is_empty:
        pieces = [leftover] if isinstance(leftover, Polygon) else list(leftover.geoms)
        hosp_pts = pts_wgs.geometry
        for piece in pieces:
            rep = piece.representative_point()
            dists = hosp_pts.distance(rep)
            nearest_idx = int(dists.idxmin())
            cur = result.loc[result['Hospital'] == hosp_names[nearest_idx], 'geometry'].values[0]
            if cur is None or cur.is_empty:
                result.loc[result['Hospital'] == hosp_names[nearest_idx], 'geometry'] = [piece]
            else:
                result.loc[result['Hospital'] == hosp_names[nearest_idx], 'geometry'] = [cur.union(piece)]

    return result


def voronoi_polygons_clipped(points_gdf: gpd.GeoDataFrame, clip_gdf: gpd.GeoDataFrame, distance_cap_km: float = CATCHMENT_DISTANCE_KM) -> gpd.GeoDataFrame:
    """
    Return per-hospital Voronoi polygons clipped to clip_gdf and trimmed to distance cap (in km).
    points_gdf: GeoDataFrame with 'Hospital' and Point geometry in WGS84
    """
    if points_gdf.empty:
        return gpd.GeoDataFrame(columns=["Hospital", "geometry"], crs=CRS_WGS84)

    pts_wgs = points_gdf.reset_index(drop=True).to_crs(CRS_WGS84)
    pts_proj = points_gdf.reset_index(drop=True).to_crs(CRS_WEBMERC)

    # union of clip polygon(s)
    clip_union = clip_gdf.unary_union

    # bounding envelope in projected coords
    clip_proj = gpd.GeoSeries([clip_union], crs=CRS_WGS84).to_crs(CRS_WEBMERC)
    minx, miny, maxx, maxy = clip_proj.total_bounds
    buf = VORONOI_BUFFER_METERS
    bbox_proj = shapely_geom.box(minx - buf, miny - buf, maxx + buf, maxy + buf)

    # Try shapely voronoi if available
    if _HAS_SHAPELY_VORONOI:
        try:
            multip = shapely_geom.MultiPoint([(pt.x, pt.y) for pt in pts_proj.geometry])
            vor = voronoi_diagram(multip, envelope=bbox_proj, tolerance=0.0)

            poly_list = []
            try:
                for g in vor.geoms:
                    if isinstance(g, (Polygon, MultiPolygon)):
                        poly_list.append(g)
            except Exception:
                if isinstance(vor, (Polygon, MultiPolygon)):
                    poly_list = [vor]

            polys_proj_gdf = gpd.GeoDataFrame(geometry=poly_list, crs=CRS_WEBMERC)
            polys_wgs = polys_proj_gdf.to_crs(CRS_WGS84).reset_index(drop=True)

            # assign polygons to nearest hospital representative point
            hosp_points_wgs = pts_wgs.set_geometry(pts_wgs.geometry).copy()
            assigned = []
            for poly in polys_wgs.geometry:
                if poly is None or poly.is_empty:
                    continue
                rep = poly.representative_point()
                dists = hosp_points_wgs.geometry.distance(rep)
                nearest_idx = int(dists.idxmin())
                hosp_name = hosp_points_wgs.loc[nearest_idx, 'Hospital']
                clipped_poly = poly.intersection(clip_union)
                if clipped_poly is None or clipped_poly.is_empty:
                    continue
                assigned.append((hosp_name, clipped_poly))

            rows = []
            for hosp in pts_wgs['Hospital'].values:
                polys_for = [p for (h, p) in assigned if h == hosp]
                geom_union = unary_union(polys_for) if polys_for else Polygon()
                rows.append({'Hospital': hosp, 'geometry': geom_union})
            result = gpd.GeoDataFrame(rows, crs=CRS_WGS84)

        except Exception:
            # fallback to scipy
            result = _scipy_voronoi_fallback(pts_wgs, pts_proj, clip_union, bbox_proj)
    else:
        result = _scipy_voronoi_fallback(pts_wgs, pts_proj, clip_union, bbox_proj)

    # Apply distance cap: intersect each region with a circular buffer around the hospital (geodesic)
    geod = Geod(ellps="WGS84")
    capped_result = []
    for _, row in result.iterrows():
        hosp_name = row['Hospital']
        geom = row['geometry']
        if geom is None or geom.is_empty:
            capped_result.append({'Hospital': hosp_name, 'geometry': Polygon()})
            continue

        hosp_row = pts_wgs[pts_wgs['Hospital'] == hosp_name]
        if hosp_row.empty:
            capped_result.append({'Hospital': hosp_name, 'geometry': Polygon()})
            continue

        hosp_point = hosp_row.geometry.iloc[0]
        hosp_lon = hosp_point.x
        hosp_lat = hosp_point.y

        # build geodesic circle
        angles = np.linspace(0, 360, 128)
        circle_points = []
        for angle in angles:
            lon2, lat2, _ = geod.fwd(hosp_lon, hosp_lat, angle, distance_cap_km * 1000)
            circle_points.append((lon2, lat2))
        circle_poly = Polygon(circle_points)

        capped_geom = geom.intersection(circle_poly)
        if capped_geom.is_empty:
            capped_geom = Polygon()
        elif isinstance(capped_geom, MultiPolygon):
            capped_geom = max(capped_geom.geoms, key=lambda p: p.area)

        capped_result.append({'Hospital': hosp_name, 'geometry': capped_geom})

    return gpd.GeoDataFrame(capped_result, crs=CRS_WGS84)


# -------------------------
# Area calculation
# -------------------------
def calculate_catchment_area(gaza_gdf: gpd.GeoDataFrame, hospitals_gdf: gpd.GeoDataFrame, hospital_intervals: dict, period_start: datetime, period_end: datetime) -> Dict[str, float]:
    """
    Returns mapping hospital_name -> catchment area (km^2) for the given period.
    Handles sub-intervals if hospitals change status inside the period (weighted by time).
    Only includes hospitals present in hospitals_gdf (and that match TARGET_HOSPITALS).
    """
    geod = Geod(ellps="WGS84")
    change_dates = []
    for hosp_name in hospitals_gdf['Hospital'].values:
        if hosp_name in hospital_intervals:
            for dt, _ in hospital_intervals[hosp_name]:
                if period_start <= dt < period_end:
                    change_dates.append(dt)

    # include period boundaries and sort unique
    change_dates = sorted(set([period_start] + change_dates + [period_end]))
    total_duration = (period_end - period_start).total_seconds()
    catchment_areas = {hosp: 0.0 for hosp in hospitals_gdf['Hospital'].values}

    for i in range(len(change_dates) - 1):
        sub_start = change_dates[i]
        sub_end = change_dates[i + 1]
        sub_duration = (sub_end - sub_start).total_seconds()
        weight = sub_duration / total_duration if total_duration > 0 else 0.0

        sub_midpoint = sub_start + (sub_end - sub_start) / 2
        open_hospitals_rows = []
        for _, row in hospitals_gdf.iterrows():
            hosp_name = row['Hospital']
            canonical = match_hospital_name(hosp_name)
            if canonical and canonical in TARGET_HOSPITALS:
                status = get_hospital_status_at_date(hospital_intervals, hosp_name, sub_midpoint)
                if status == "Open":
                    open_hospitals_rows.append({'Hospital': hosp_name, 'geometry': row.geometry})

        if not open_hospitals_rows:
            continue

        open_hospitals_gdf = gpd.GeoDataFrame(open_hospitals_rows, crs=CRS_WGS84)
        voronoi_polys = voronoi_polygons_clipped(open_hospitals_gdf, gaza_gdf, distance_cap_km=CATCHMENT_DISTANCE_KM)

        for _, row in voronoi_polys.iterrows():
            hosp_name = row['Hospital']
            geom = row['geometry']
            if geom is None or geom.is_empty:
                continue
            try:
                if isinstance(geom, MultiPolygon):
                    total_area_m2 = 0
                    for poly in geom.geoms:
                        area_m2, _ = geod.geometry_area_perimeter(poly)
                        total_area_m2 += abs(area_m2)
                else:
                    area_m2, _ = geod.geometry_area_perimeter(geom)
                    total_area_m2 = abs(area_m2)
                area_km2 = total_area_m2 / 1e6
                catchment_areas[hosp_name] += area_km2 * weight
            except Exception:
                # skip problem polygons but continue
                continue

    return catchment_areas


# -------------------------
# Visualization
# -------------------------
def create_html_visualization(gaza_gdf: gpd.GeoDataFrame, hospitals_gdf: gpd.GeoDataFrame, hospital_intervals: dict, period_start: datetime, period_end: datetime, output_path: str):
    """
    Create folium HTML visualization for the given period (uses midpoint statuses).
    """
    sub_midpoint = period_start + (period_end - period_start) / 2
    open_hospitals_rows = []
    hosp_coords = {}

    for _, row in hospitals_gdf.iterrows():
        hosp_name = row['Hospital']
        canonical = match_hospital_name(hosp_name)
        if canonical and canonical in TARGET_HOSPITALS:
            status = get_hospital_status_at_date(hospital_intervals, hosp_name, sub_midpoint)
            if status == "Open":
                open_hospitals_rows.append({'Hospital': hosp_name, 'geometry': row.geometry})
                hosp_coords[hosp_name] = (row.geometry.y, row.geometry.x)

    if not open_hospitals_rows:
        print(f"No open target hospitals in period {period_start} to {period_end} — skipping HTML creation.")
        return

    open_hospitals_gdf = gpd.GeoDataFrame(open_hospitals_rows, crs=CRS_WGS84)
    voronoi_polys = voronoi_polygons_clipped(open_hospitals_gdf, gaza_gdf, distance_cap_km=CATCHMENT_DISTANCE_KM)

    bounds = gaza_gdf.total_bounds  # minx, miny, maxx, maxy in WGS84
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    m = folium.Map(location=center, zoom_start=11)

    # Add polygons with hospital-specific colors
    for _, row in voronoi_polys.iterrows():
        hosp_name = row['Hospital']
        canonical = match_hospital_name(hosp_name)
        geom = row['geometry']
        if geom is None or geom.is_empty:
            continue
        if canonical:
            color = HOSPITAL_COLORS.get(canonical, "#999999")
        else:
            color = "#999999"

        folium.GeoJson(
            geom.__geo_interface__,
            style_function=lambda feature, fill_color=color: {
                'fillColor': fill_color,
                'fillOpacity': 0.5,
                'color': 'black',
                'weight': 1,
                'opacity': 0.8
            },
            tooltip=hosp_name
        ).add_to(m)

    # hospital markers
    for hosp_name, (lat, lon) in hosp_coords.items():
        canonical = match_hospital_name(hosp_name)
        folium.Marker(
            location=(lat, lon),
            popup=hosp_name,
            icon=folium.Icon(color='red', icon='plus-sign', prefix='glyphicon')
        ).add_to(m)

    # Gaza boundary
    folium.GeoJson(
        gaza_gdf.__geo_interface__,
        style_function=lambda feature: {
            'fillColor': 'transparent',
            'color': 'black',
            'weight': 2,
            'opacity': 1.0
        }
    ).add_to(m)

    # Title box
    open_names = [n for n in open_hospitals_gdf['Hospital'].values]
    title_html = f"""
    <div style="position: fixed; top: 10px; left: 10px; z-index: 9999;
                background: white; padding: 8px; border: 1px solid grey; font-size:12px;">
        <b>Catchment Areas</b><br/>
        <b>Period:</b> {period_start.strftime('%Y-%m-%d')} → {period_end.strftime('%Y-%m-%d')}<br/>
        <b>Open hospitals:</b> {', '.join(open_names)}
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    # Legend (programmatically generated)
    legend_items = []
    for canonical in TARGET_HOSPITALS:
        color = HOSPITAL_COLORS.get(canonical, "#999999")
        legend_items.append(f'<div style="margin-bottom:4px;"><span style="display:inline-block;width:12px;height:12px;background:{color};margin-right:6px;border:1px solid #000;"></span>{canonical}</div>')
    legend_html = f"""
    <div style="position: fixed; bottom: 30px; left: 10px; z-index:9999;
                background: white; padding: 8px; border: 1px solid grey; font-size:12px;">
        <b>Hospital Catchment Areas</b><br/>
        {''.join(legend_items)}
        <div style="margin-top:6px;font-size:10px;">Areas within {CATCHMENT_DISTANCE_KM} km of closest open hospital</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Save
    m.save(output_path)
    print(f"Saved HTML visualization to: {output_path}")


# -------------------------
# Main orchestration
# -------------------------
def main():
    print("=" * 70)
    print("Catchment Area Calculations (rewritten)")
    print("=" * 70)

    # Read hospitals table
    hospitals_df, schedule_meta = read_hospitals_table(HOSP_PATH)
    print(f"Loaded {len(hospitals_df)} hospital records from Excel.")

    # Show simple mapping diagnostics
    matched = []
    for hosp_name in hospitals_df['Hospital'].values:
        canonical = match_hospital_name(hosp_name)
        matched.append((hosp_name, canonical))
    print("Detected mapping (Excel name -> canonical):")
    for orig, canon in matched:
        print(f"  '{orig}' -> {canon}")

    # Build hospital intervals
    hospital_intervals = build_hospital_open_intervals(hospitals_df, schedule_meta)

    # Determine date range
    all_dates = [d for (_, _, d) in schedule_meta if d is not None]
    # also scan per-row cells for dates (robust)
    for _, row in hospitals_df.iterrows():
        for col, _, dt in schedule_meta:
            if col in row.index and pd.notnull(row.get(col)):
                try:
                    cell_date = pd.to_datetime(row[col])
                    if pd.notnull(cell_date):
                        all_dates.append(cell_date.to_pydatetime())
                except Exception:
                    pass

    if not all_dates:
        raise ValueError("Could not determine date range from Excel. No dates found in schedule header or cells.")

    START_DATE = min(all_dates)
    END_DATE = max(all_dates)
    print(f"Date range discovered: {START_DATE.date()} to {END_DATE.date()}")

    # create hospitals GeoDataFrame (only rows with valid coords)
    hospitals_gdf = gpd.GeoDataFrame(
        hospitals_df.dropna(subset=["lon", "lat"]).copy(),
        geometry=gpd.points_from_xy(hospitals_df["lon"].astype(float), hospitals_df["lat"].astype(float)),
        crs=CRS_WGS84
    )

    # load Gaza boundary
    if not os.path.exists(GAZA_BOUNDARY):
        raise FileNotFoundError(f"Gaza boundary file not found: {GAZA_BOUNDARY}")
    gaza_gdf = gpd.read_file(GAZA_BOUNDARY)
    if gaza_gdf.crs is None:
        gaza_gdf = gaza_gdf.set_crs(CRS_WGS84)
    gaza_gdf = gaza_gdf.to_crs(CRS_WGS84)
    gaza_gdf = gaza_gdf.dissolve(by=None).reset_index(drop=True)

    # calculate Gaza area for percent computations
    geod = Geod(ellps="WGS84")
    gaza_geom = gaza_gdf.unary_union
    if isinstance(gaza_geom, MultiPolygon):
        total_gaza_area_m2 = sum(abs(geod.geometry_area_perimeter(poly)[0]) for poly in gaza_geom.geoms)
    else:
        total_gaza_area_m2 = abs(geod.geometry_area_perimeter(gaza_geom)[0])
    total_gaza_area_km2 = total_gaza_area_m2 / 1e6
    print(f"Total Gaza area: {total_gaza_area_km2:.2f} km²")

    # iterate two-week periods
    print("Calculating catchment areas in two-week increments...")
    results = []
    current_date = START_DATE
    period_num = 1

    while current_date < END_DATE:
        period_start = current_date
        period_end = min(current_date + timedelta(days=14), END_DATE)
        print(f"\nPeriod {period_num}: {period_start.date()} to {period_end.date()}")
        catchment_areas = calculate_catchment_area(gaza_gdf, hospitals_gdf, hospital_intervals, period_start, period_end)

        # store results for all hospitals that match target list
        row = {"Period": f"{period_start.date()} to {period_end.date()}"}
        for _, hrow in hospitals_gdf.iterrows():
            hosp_name = hrow['Hospital']
            canonical = match_hospital_name(hosp_name)
            if canonical and canonical in TARGET_HOSPITALS:
                row[hosp_name] = catchment_areas.get(hosp_name, 0.0)
        results.append(row)

        # create HTML for first period
        if period_num == 1:
            safe_name = f"catchment_visualization_{period_start.strftime('%Y%m%d')}_to_{period_end.strftime('%Y%m%d')}.html"
            html_path = os.path.join(OUTPUT_DIR, safe_name)
            create_html_visualization(gaza_gdf, hospitals_gdf, hospital_intervals, period_start, period_end, html_path)

        current_date = period_end
        period_num += 1

    # prepare Excel output (area rows and percentage rows)
    print("\nPreparing Excel output...")
    all_hospital_names = sorted({k for r in results for k in r.keys() if k != "Period"})
    period_cols = [r["Period"] for r in results]

    excel_rows = []
    for hosp_name in all_hospital_names:
        canonical = match_hospital_name(hosp_name)
        if not canonical or canonical not in TARGET_HOSPITALS:
            continue

        # area row
        area_row = {"Hospital": hosp_name}
        area_values = []
        for rr in results:
            period = rr["Period"]
            area = rr.get(hosp_name, 0.0)
            area_row[period] = area
            area_values.append(area)
        area_row["Average"] = float(np.mean(area_values)) if area_values else 0.0
        excel_rows.append(area_row)

        # percent row
        pct_row = {"Hospital": ""}
        for rr in results:
            period = rr["Period"]
            area = rr.get(hosp_name, 0.0)
            pct = (area / total_gaza_area_km2 * 100.0) if total_gaza_area_km2 > 0 else 0.0
            pct_row[period] = f"{pct:.2f}%"
        avg_pct = (np.mean(area_values) / total_gaza_area_km2 * 100.0) if total_gaza_area_km2 > 0 else 0.0
        pct_row["Average"] = f"{avg_pct:.2f}%"
        excel_rows.append(pct_row)

    if excel_rows:
        df_out = pd.DataFrame(excel_rows)
        cols = ["Hospital"] + period_cols + ["Average"]
        df_out = df_out[[c for c in cols if c in df_out.columns]]

        excel_path = os.path.join(OUTPUT_DIR, "catchment_areas_over_time.xlsx")
        df_out.to_excel(excel_path, index=False)
        print(f"Saved Excel file to: {excel_path}")
        print("\nSample of results:")
        with pd.option_context('display.max_rows', 200, 'display.max_columns', 200):
            print(df_out.head(20).to_string(index=False))
    else:
        print("No hospital data to write to Excel.")

    print("\nProcessing complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
