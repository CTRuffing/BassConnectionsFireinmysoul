#!/usr/bin/env python3
"""
catchment_calculations_full_rewrite.py

Complete, cleaned-up rewrite of your catchment-area script.

Features:
- Robust hospital-name matching (reliably includes Al Quds and Kuwait).
- Voronoi regions (shapely.voronoi_diagram when available; SciPy fallback).
- 5 km catchment cap (geodesic buffer).
- Two-week aggregates with weighting if hospital status changes mid-period.
- Outputs:
    - Excel workbook "catchment_areas_over_time.xlsx" with TWO SHEETS:
        1) "Catchment areas over time" (areas in km²)
        2) "Catchment percentages over time" (percent of total Gaza area)
    - HTML map for the first two-week period: saved as catchment_visualization_YYYYMMDD_to_YYYYMMDD.html

Drop this file in the same folder as your Excel & geojson and run:
    python catchment_calculations_full_rewrite.py
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

# try shapely voronoi (optional optimization)
try:
    from shapely.ops import voronoi_diagram  # type: ignore
    _HAS_SHAPELY_VORONOI = True
except Exception:
    voronoi_diagram = None
    _HAS_SHAPELY_VORONOI = False

# -------------------------
# Configuration
# -------------------------
HOSP_PATH = "Hospitals_OpenCloseoverTime.xlsx"
GAZA_BOUNDARY = "gaza_boundary.geojson"
OUTPUT_DIR = "."
CRS_WGS84 = "EPSG:4326"
CRS_WEBMERC = "EPSG:3857"
CATCHMENT_DISTANCE_KM = 5.0
VORONOI_BUFFER_METERS = 15000

# canonical targets and common variants
HOSPITAL_NAME_MAPPING = {
    "Al Nasser": ["Al Nasser", "Nasser Hospital", "Nasser"],
    "EGH": ["EGH", "European Hospital", "European", "El European"],
    "Al Shifa": ["Al Shifa", "Al Shifa Medical Hospital", "Shifa"],
    "Al Quds": ["Al Quds", "Al-Quds", "Al Quds Hospital", "Al-Quds Hospital"],
    "Kuwait": ["Kuwait", "Kuwait Hospital", "Kuwait Hosp"]
}
TARGET_HOSPITALS = [
    "Al Shifa",
    "EGH",
    "Al Nasser",
]

# colors for map legend
HOSPITAL_COLORS = {
    "Al Nasser": "#e41a1c",
    "EGH": "#377eb8",
    "Al Shifa": "#4daf4a",
    "Al Quds": "#984ea3",
    "Kuwait": "#ff7f00"
}


# -------------------------
# Utility / matching
# -------------------------
def _normalize(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s)
    s2 = re.sub(r"[^0-9a-zA-Z]+", " ", s).strip().lower()
    s2 = re.sub(r"\s+", " ", s2)
    return s2


def match_hospital_name(name: Optional[str]) -> Optional[str]:
    """
    Robustly map various hospital name strings to canonical names in HOSPITAL_NAME_MAPPING.
    Returns canonical name (e.g., "Al Quds") or None.
    """
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return None
    name_norm = _normalize(name)
    if not name_norm:
        return None

    # exact normalized match
    for canonical, variants in HOSPITAL_NAME_MAPPING.items():
        for v in variants + [canonical]:
            if _normalize(v) == name_norm:
                return canonical

    # token overlap (ignore stop tokens)
    name_tokens = set(name_norm.split())
    stop_tokens = {"al", "the", "hospital", "hosp", "medical"}
    for canonical, variants in HOSPITAL_NAME_MAPPING.items():
        tokens = set(_normalize(canonical).split())
        for v in variants:
            tokens.update(_normalize(v).split())
        sig = [t for t in (tokens & name_tokens) if t not in stop_tokens]
        if sig:
            return canonical

    # fallback heuristics
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
    Returns DataFrame with 'Hospital','lon','lat' and a schedule_meta list of (col, type, date).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Hospitals table not found at: {path}")

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

    # keep only schedule columns that had a date in the second header row
    schedule_meta = [(c, typ, d) for (c, typ, d) in schedule_meta if d is not None]

    hospitals_df = full[[hosp_col, lon_col, lat_col]].rename(columns={hosp_col: "Hospital", lon_col: "lon", lat_col: "lat"})
    hospitals_df["lon"] = pd.to_numeric(hospitals_df["lon"], errors="coerce")
    hospitals_df["lat"] = pd.to_numeric(hospitals_df["lat"], errors="coerce")

    # copy schedule marker columns if they exist
    for c, _, _ in schedule_meta:
        if c in full.columns:
            hospitals_df[c] = full[c]

    return hospitals_df, schedule_meta


def build_hospital_open_intervals(hospitals_df: pd.DataFrame, schedule_meta: List[Tuple[str, str, Optional[datetime]]]) -> Dict[str, List[Tuple[datetime, str]]]:
    """
    Build per-hospital ordered list of (date, status) changes.
    If row-specific markers are present they are used, otherwise global schedule events are applied.
    """
    events = sorted([(pd.to_datetime(d).to_pydatetime(), col, typ) for (col, typ, d) in schedule_meta], key=lambda x: x[0])
    hospital_intervals: Dict[str, List[Tuple[datetime, str]]] = {}

    for _, row in hospitals_df.iterrows():
        name = row["Hospital"]
        changes: List[Tuple[datetime, str]] = []

        # per-row markers (preferred)
        for dt, col, typ in events:
            if col in row.index:
                val = row.get(col, None)
                if pd.notnull(val) and str(val).strip() != "":
                    changes.append((dt, typ))

        # if none found per-row, apply global events
        if not changes:
            for dt, col, typ in events:
                changes.append((dt, typ))

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
    Default to "Open" if no prior event exists.
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
# Voronoi & clipping
# -------------------------
def _scipy_voronoi_fallback(pts_wgs: gpd.GeoDataFrame, pts_proj: gpd.GeoDataFrame, clip_union, bbox_proj) -> gpd.GeoDataFrame:
    coords = np.array([(pt.x, pt.y) for pt in pts_proj.geometry])
    vor = Voronoi(coords)

    proj_polys = []
    for pt_idx, region_index in enumerate(vor.point_region):
        region = vor.regions[region_index]
        if not region or -1 in region:
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

    # assign leftover pieces to nearest hospital
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
    Build Voronoi regions for point hospitals, clip them to clip_gdf (WGS84),
    then intersect each with a geodesic circle (distance_cap_km).
    Returns GeoDataFrame with columns ['Hospital','geometry'] in WGS84.
    """
    if points_gdf.empty:
        return gpd.GeoDataFrame(columns=["Hospital", "geometry"], crs=CRS_WGS84)

    pts_wgs = points_gdf.reset_index(drop=True).to_crs(CRS_WGS84)
    pts_proj = points_gdf.reset_index(drop=True).to_crs(CRS_WEBMERC)

    clip_union = clip_gdf.unary_union
    clip_proj = gpd.GeoSeries([clip_union], crs=CRS_WGS84).to_crs(CRS_WEBMERC)
    minx, miny, maxx, maxy = clip_proj.total_bounds
    buf = VORONOI_BUFFER_METERS
    bbox_proj = shapely_geom.box(minx - buf, miny - buf, maxx + buf, maxy + buf)

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
            result = _scipy_voronoi_fallback(pts_wgs, pts_proj, clip_union, bbox_proj)
    else:
        result = _scipy_voronoi_fallback(pts_wgs, pts_proj, clip_union, bbox_proj)

    # distance cap (geodesic circle around each hospital)
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

        # create geodesic circle
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
    Return mapping hospital_name -> catchment area (km^2) for the period.
    Handles sub-intervals if hospitals change status mid-period (weights by time).
    """
    geod = Geod(ellps="WGS84")
    change_dates = []
    for hosp_name in hospitals_gdf['Hospital'].values:
        if hosp_name in hospital_intervals:
            for dt, _ in hospital_intervals[hosp_name]:
                if period_start <= dt < period_end:
                    change_dates.append(dt)

    change_dates = sorted(set([period_start] + change_dates + [period_end]))
    total_duration = (period_end - period_start).total_seconds()
    catchment_areas = {hosp: 0.0 for hosp in hospitals_gdf['Hospital'].values}

    for i in range(len(change_dates) - 1):
        sub_start = change_dates[i]
        sub_end = change_dates[i + 1]
        sub_duration = (sub_end - sub_start).total_seconds()
        weight = sub_duration / total_duration if total_duration > 0 else 0.0

        sub_midpoint = sub_start + (sub_end - sub_start) / 2
        open_rows = []
        for _, row in hospitals_gdf.iterrows():
            hosp_name = row['Hospital']
            canonical = match_hospital_name(hosp_name)
            if canonical and canonical in TARGET_HOSPITALS:
                status = get_hospital_status_at_date(hospital_intervals, hosp_name, sub_midpoint)
                if status == "Open":
                    open_rows.append({'Hospital': hosp_name, 'geometry': row.geometry})

        if not open_rows:
            continue

        open_gdf = gpd.GeoDataFrame(open_rows, crs=CRS_WGS84)
        voronoi_polys = voronoi_polygons_clipped(open_gdf, gaza_gdf, distance_cap_km=CATCHMENT_DISTANCE_KM)

        for _, r in voronoi_polys.iterrows():
            hosp_name = r['Hospital']
            geom = r['geometry']
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
                continue

    return catchment_areas


# -------------------------
# Visualization
# -------------------------
def create_html_visualization(gaza_gdf: gpd.GeoDataFrame, hospitals_gdf: gpd.GeoDataFrame, hospital_intervals: dict, period_start: datetime, period_end: datetime, output_path: str):
    """
    Create folium HTML visualization for first period (or any chosen period).
    """
    sub_midpoint = period_start + (period_end - period_start) / 2
    open_rows = []
    coords = {}

    for _, row in hospitals_gdf.iterrows():
        hosp_name = row['Hospital']
        canonical = match_hospital_name(hosp_name)
        if canonical and canonical in TARGET_HOSPITALS:
            status = get_hospital_status_at_date(hospital_intervals, hosp_name, sub_midpoint)
            if status == "Open":
                open_rows.append({'Hospital': hosp_name, 'geometry': row.geometry})
                coords[hosp_name] = (row.geometry.y, row.geometry.x)

    if not open_rows:
        print(f"No open target hospitals in period {period_start} to {period_end} — skipping HTML.")
        return

    open_gdf = gpd.GeoDataFrame(open_rows, crs=CRS_WGS84)
    voronoi_polys = voronoi_polygons_clipped(open_gdf, gaza_gdf, distance_cap_km=CATCHMENT_DISTANCE_KM)

    bounds = gaza_gdf.total_bounds  # minx, miny, maxx, maxy
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    m = folium.Map(location=center, zoom_start=11)

    for _, r in voronoi_polys.iterrows():
        hosp_name = r['Hospital']
        canonical = match_hospital_name(hosp_name)
        geom = r['geometry']
        if geom is None or geom.is_empty:
            continue
        color = HOSPITAL_COLORS.get(canonical, "#999999") if canonical else "#999999"
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

    for hosp_name, (lat, lon) in coords.items():
        folium.Marker(location=(lat, lon), popup=hosp_name, icon=folium.Icon(color='red', icon='plus-sign', prefix='glyphicon')).add_to(m)

    folium.GeoJson(gaza_gdf.__geo_interface__, style_function=lambda f: {'fillColor': 'transparent', 'color': 'black', 'weight': 2}).add_to(m)

    open_names = [n for n in open_gdf['Hospital'].values]
    title_html = f"""
    <div style="position: fixed; top: 10px; left: 10px; z-index: 9999; background: white; padding: 8px; border:1px solid grey; font-size:12px;">
      <b>Catchment Areas</b><br/>
      <b>Period:</b> {period_start.strftime('%Y-%m-%d')} → {period_end.strftime('%Y-%m-%d')}<br/>
      <b>Open hospitals:</b> {', '.join(open_names)}
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    legend_items = []
    for canonical in TARGET_HOSPITALS:
        color = HOSPITAL_COLORS.get(canonical, "#999999")
        legend_items.append(f'<div style="margin-bottom:4px;"><span style="display:inline-block;width:12px;height:12px;background:{color};margin-right:6px;border:1px solid #000;"></span>{canonical}</div>')
    legend_html = f"""
    <div style="position: fixed; bottom: 30px; left: 10px; z-index:9999; background: white; padding: 8px; border:1px solid grey; font-size:12px;">
      <b>Hospital Catchment Areas</b><br/>{''.join(legend_items)}
      <div style="margin-top:6px;font-size:10px;">Areas within {CATCHMENT_DISTANCE_KM} km of closest open hospital</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(output_path)
    print(f"Saved HTML visualization to: {output_path}")


# -------------------------
# Main
# -------------------------
def main():
    print("=" * 70)
    print("Catchment Area Calculations — Full Rewrite")
    print("=" * 70)

    # read hospitals & schedule
    hospitals_df, schedule_meta = read_hospitals_table(HOSP_PATH)
    print(f"Loaded {len(hospitals_df)} hospital records.")

    # diagnostics: show mapping
    print("Hospital name mapping (excel name -> canonical):")
    for hosp_name in hospitals_df['Hospital'].values:
        print(f"  '{hosp_name}' -> {match_hospital_name(hosp_name)}")

    hospital_intervals = build_hospital_open_intervals(hospitals_df, schedule_meta)

    # OVERRIDE: force hospital timelines to the user-specified ranges
    # Ignore the open/close spreadsheet and apply explicit intervals for only the
    # three target hospitals (Al Shifa, EGH, Al Nasser). We store intervals
    # keyed by the Excel 'Hospital' name so downstream functions that look up
    # hospital status by that name will find these entries.
    explicit_ranges = {
        "Al Shifa": (datetime(2023, 10, 7), datetime(2023, 11, 3)),
        "EGH": (datetime(2023, 12, 11), datetime(2024, 4, 28)),
        "Al Nasser": (datetime(2024, 11, 11), datetime(2025, 2, 2)),
    }

    # Build override mapping for hospital_intervals using the Excel hospital names
    # that map to the canonical targets.
    override_intervals = {}
    for hosp_name in hospitals_df['Hospital'].values:
        canonical = match_hospital_name(hosp_name)
        if canonical in TARGET_HOSPITALS:
            # ensure closed before start, open at start, closed after end
            start, end = explicit_ranges.get(canonical, (None, None))
            if start is not None and end is not None:
                override_intervals[hosp_name] = [
                    (datetime(1900, 1, 1), "Closed"),
                    (start, "Open"),
                    (end + timedelta(days=1), "Closed"),
                ]

    # Replace hospital_intervals entries for matching hospitals
    for k, v in override_intervals.items():
        hospital_intervals[k] = v

    # determine date range (kept for diagnostics / compatibility)
    all_dates = [d for (_, _, d) in schedule_meta if d is not None]
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
        raise ValueError("Could not determine date range from Excel schedule headers/cells.")

    # original min/max (still useful to know overall bounds)
    START_DATE = min(all_dates)
    END_DATE = max(all_dates)
    print(f"Overall schedule date range (from spreadsheet): {START_DATE.date()} to {END_DATE.date()}")

    # create hospitals GeoDataFrame
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

    # compute Gaza area for percentages
    geod = Geod(ellps="WGS84")
    gaza_geom = gaza_gdf.unary_union
    if isinstance(gaza_geom, MultiPolygon):
        total_gaza_area_m2 = sum(abs(geod.geometry_area_perimeter(poly)[0]) for poly in gaza_geom.geoms)
    else:
        total_gaza_area_m2 = abs(geod.geometry_area_perimeter(gaza_geom)[0])
    total_gaza_area_km2 = total_gaza_area_m2 / 1e6
    print(f"Total Gaza area: {total_gaza_area_km2:.2f} km²")

    # -------------------------
    # Build two-week periods INSIDE the user-specified ranges
    # -------------------------
    date_ranges = [
        (datetime(2023, 10, 7), datetime(2023, 11, 3)),
        (datetime(2023, 12, 11), datetime(2024, 4, 28)),
        (datetime(2024, 11, 11), datetime(2025, 2, 2)),
    ]

    # normalize ranges (ensure start < end)
    norm_ranges = []
    for a, b in date_ranges:
        if a >= b:
            continue
        norm_ranges.append((a, b))

    periods = []
    two_weeks = timedelta(days=13)
    for (rstart, rend) in norm_ranges:
        cur = rstart
        while cur < rend:
            period_start = cur
            period_end = min(cur + two_weeks, rend)
            periods.append((period_start, period_end))
            cur = period_end + timedelta(days = 1) # jump by each two-week chunk inside this range

    if not periods:
        raise ValueError("No periods were generated from the provided date_ranges.")

    print(f"Generated {len(periods)} two-week periods from provided ranges.")
    for i, (ps, pe) in enumerate(periods, 1):
        print(f"  Period {i}: {ps.date()} to {pe.date()}")

    # iterate periods and compute catchments
    print("Calculating catchment areas for defined two-week periods...")
    results = []
    period_num = 1

    for (period_start, period_end) in periods:
        print(f"\nPeriod {period_num}: {period_start.date()} to {period_end.date()}")

        catchment_areas = calculate_catchment_area(gaza_gdf, hospitals_gdf, hospital_intervals, period_start, period_end)

        row = {"Period": f"{period_start.date()} to {period_end.date()}"}
        for _, hrow in hospitals_gdf.iterrows():
            hosp_name = hrow['Hospital']
            canonical = match_hospital_name(hosp_name)
            if canonical and canonical in TARGET_HOSPITALS:
                row[hosp_name] = catchment_areas.get(hosp_name, 0.0)
        results.append(row)

        # save HTML for the first processed period only
        if period_num == 1:
            html_name = f"catchment_visualization_{period_start.strftime('%Y%m%d')}_to_{period_end.strftime('%Y%m%d')}.html"
            html_path = os.path.join(OUTPUT_DIR, html_name)
            create_html_visualization(gaza_gdf, hospitals_gdf, hospital_intervals, period_start, period_end, html_path)

        period_num += 1

    # Prepare Excel output with TWO sheets:
    #   "Catchment areas over time" (km²)
    #   "Catchment percentages over time" (% of Gaza area)
    print("\nPreparing Excel workbook with two sheets...")
    all_hospital_names = sorted({k for r in results for k in r.keys() if k != "Period"})
    period_cols = [r["Period"] for r in results]

    areas_rows = []
    pct_rows = []

    for hosp_name in all_hospital_names:
        canonical = match_hospital_name(hosp_name)
        if not canonical or canonical not in TARGET_HOSPITALS:
            continue

        area_row = {"Hospital": hosp_name}
        area_vals = []
        for rr in results:
            period = rr["Period"]
            area = rr.get(hosp_name, 0.0)
            area_row[period] = area
            area_vals.append(area)
        area_row["Average"] = float(np.mean(area_vals)) if area_vals else 0.0
        areas_rows.append(area_row)

        pct_row = {"Hospital": hosp_name}
        for rr in results:
            period = rr["Period"]
            area = rr.get(hosp_name, 0.0)
            pct = (area / total_gaza_area_km2 * 100.0) if total_gaza_area_km2 > 0 else 0.0
            pct_row[period] = pct
        avg_pct = (np.mean(area_vals) / total_gaza_area_km2 * 100.0) if total_gaza_area_km2 > 0 else 0.0
        pct_row["Average"] = avg_pct
        pct_rows.append(pct_row)

    if areas_rows or pct_rows:
        areas_df = pd.DataFrame(areas_rows)
        pct_df = pd.DataFrame(pct_rows)

        cols = ["Hospital"] + period_cols + ["Average"]
        areas_df = areas_df[[c for c in cols if c in areas_df.columns]]
        pct_df = pct_df[[c for c in cols if c in pct_df.columns]]

        excel_path = os.path.join(OUTPUT_DIR, "catchment_areas_over_time.xlsx")
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            areas_df.to_excel(writer, sheet_name="Catchment areas over time", index=False)
            pct_df.to_excel(writer, sheet_name="Catchment percentages over time", index=False)

        print(f"Saved workbook: {excel_path}")
        print("\nSample (areas):")
        with pd.option_context('display.max_rows', 200, 'display.max_columns', 200):
            print(areas_df.head(20).to_string(index=False))
        print("\nSample (percentages):")
        with pd.option_context('display.max_rows', 200, 'display.max_columns', 200):
            print(pct_df.head(20).to_string(index=False))
    else:
        print("No data to write to Excel.")

    print("\nDone.")
    print("=" * 70)

if __name__ == "__main__":
    main()
