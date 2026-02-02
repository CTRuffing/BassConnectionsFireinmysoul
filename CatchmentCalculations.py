"""
CatchmentCalculations.py

Calculate catchment areas for hospitals (Al Nasser, EGH, and Al Shifa) over time
in two-week aggregates. Catchment area is determined by finding the closest hospital
for each point in Gaza territory (within 5km cap).

Outputs:
1. Excel file with hospitals in column A and catchment areas over time
2. HTML visualization for the first two-week period
"""

import os
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon, box
from shapely.ops import unary_union
from shapely import geometry as shapely_geom
from pyproj import Geod
from scipy.spatial import Voronoi
import folium
import warnings
warnings.filterwarnings("ignore")

# Try to import shapely voronoi_diagram
try:
    from shapely.ops import voronoi_diagram
    _HAS_SHAPELY_VORONOI = True
except Exception:
    voronoi_diagram = None
    _HAS_SHAPELY_VORONOI = False

# Configuration
HOSP_PATH = "Hospitals_OpenCloseoverTime.xlsx"
GAZA_BOUNDARY = "gaza_boundary.geojson"
OUTPUT_DIR = "."

# Target hospitals
TARGET_HOSPITALS = ["Al Nasser", "EGH", "Al Shifa"]
# Hospital name variations to match
HOSPITAL_NAME_MAPPING = {
    "Al Nasser": ["Al Nasser", "Nasser Hospital", "Nasser"],
    "EGH": ["EGH", "European Hospital", "European"],
    "Al Shifa": ["Al Shifa", "Al Shifa Medical Hospital", "Shifa"]
}

CRS_WGS84 = "EPSG:4326"
CRS_WEBMERC = "EPSG:3857"
CATCHMENT_DISTANCE_KM = 5.0  # 5 km cap
VORONOI_BUFFER_METERS = 15000  # buffer for bounding box in projected coords

# Variables for date range (will be set from Excel file)
START_DATE = None
END_DATE = None


def read_hospitals_table(path):
    """
    Parse the Hospitals_OpenCloseoverTime.xlsx file.
    Returns hospitals_df with columns ['Hospital','lon','lat'] and schedule_meta.
    """
    raw_head = pd.read_excel(path, header=None, nrows=2)
    full = pd.read_excel(path, header=0)
    cols = list(full.columns)

    def find_col(opts):
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

    # detect Open/Closed header columns and try to read date from second row
    schedule_meta = []
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

    # keep only entries which have a date
    schedule_meta = [(c, t, d) for (c, t, d) in schedule_meta if d is not None]

    hospitals_df = full[[hosp_col, lon_col, lat_col]].rename(
        columns={hosp_col: "Hospital", lon_col: "lon", lat_col: "lat"}
    )
    hospitals_df["lon"] = pd.to_numeric(hospitals_df["lon"], errors="coerce")
    hospitals_df["lat"] = pd.to_numeric(hospitals_df["lat"], errors="coerce")
    
    # copy schedule marker columns
    for c, _, _ in schedule_meta:
        if c in full.columns:
            hospitals_df[c] = full[c]

    return hospitals_df, schedule_meta


def build_hospital_open_intervals(hospitals_df, schedule_meta):
    """
    Build per-hospital ordered list of (date, status) changes.
    Assumes hospitals start as "Open" if no initial status is specified.
    """
    events = sorted(
        [(pd.to_datetime(d).to_pydatetime(), col, typ) for (col, typ, d) in schedule_meta],
        key=lambda x: x[0]
    )
    hospital_intervals = {}
    
    for _, row in hospitals_df.iterrows():
        name = row["Hospital"]
        changes = []
        
        # If per-row markers exist, use them
        for dt, col, typ in events:
            col_name = col
            val = row.get(col_name, None) if col_name in row.index else None
            if pd.notnull(val) and str(val).strip() != "":
                changes.append((dt, typ))
        
        # If none, fall back to global events
        if not changes:
            for dt, col, typ in events:
                changes.append((dt, typ))
        
        # Sort and compress consecutive duplicates
        changes_sorted = sorted(changes, key=lambda x: x[0])
        compressed = []
        for dt, typ in changes_sorted:
            if not compressed or compressed[-1][1] != typ:
                compressed.append((dt, typ))
        
        # If no changes, assume always open
        if not compressed:
            compressed = [(datetime(1900, 1, 1), "Open")]
        
        hospital_intervals[name] = compressed
    
    return hospital_intervals


def get_hospital_status_at_date(hospital_intervals, hospital_name, date):
    """
    Get the status of a hospital at a given date.
    Returns "Open" or "Closed".
    """
    if hospital_name not in hospital_intervals:
        return "Closed"
    
    changes = hospital_intervals[hospital_name]
    last_status = "Open"  # Default to open if no changes before this date
    
    for dt, typ in changes:
        if dt <= date:
            last_status = typ
        else:
            break
    
    return last_status


def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate haversine distance between two points in kilometers.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371.0 * c
    return km


def voronoi_polygons_clipped(points_gdf, clip_gdf, distance_cap_km=CATCHMENT_DISTANCE_KM):
    """
    Build Voronoi regions for points_gdf and clip them to clip_gdf (WGS84).
    Also applies distance cap - only includes areas within distance_cap_km of hospitals.
    Returns GeoDataFrame with ['geometry','Hospital'] in WGS84.
    """
    if points_gdf.empty:
        return gpd.GeoDataFrame(columns=["geometry", "Hospital"], crs=CRS_WGS84)
    
    # Prepare geometries
    pts_wgs = points_gdf.reset_index(drop=True).to_crs(CRS_WGS84)
    pts_proj = points_gdf.reset_index(drop=True).to_crs(CRS_WEBMERC)
    
    # Union clip polygon
    clip_union = clip_gdf.unary_union
    
    # Build bounding envelope in projected coordinates
    clip_proj = gpd.GeoSeries([clip_union], crs=CRS_WGS84).to_crs(CRS_WEBMERC)
    minx, miny, maxx, maxy = clip_proj.total_bounds
    buf = VORONOI_BUFFER_METERS
    bbox_proj = shapely_geom.box(minx - buf, miny - buf, maxx + buf, maxy + buf)
    
    # Try shapely.voronoi_diagram if available
    if _HAS_SHAPELY_VORONOI:
        try:
            multip = shapely_geom.MultiPoint([(pt.x, pt.y) for pt in pts_proj.geometry])
            vor = voronoi_diagram(multip, envelope=bbox_proj, tolerance=0.0)
            
            # Extract polygons
            poly_list = []
            if not vor.is_empty:
                try:
                    for g in vor.geoms:
                        if isinstance(g, (Polygon, MultiPolygon)):
                            poly_list.append(g)
                except Exception:
                    if isinstance(vor, (Polygon, MultiPolygon)):
                        poly_list = [vor]
            
            # Assign polygons to nearest hospital
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
            
            # Build result
            rows = []
            for hosp in pts_wgs['Hospital'].values:
                polys_for = [p for (h, p) in assigned if h == hosp]
                geom_union = unary_union(polys_for) if polys_for else Polygon()
                rows.append({'Hospital': hosp, 'geometry': geom_union})
            result = gpd.GeoDataFrame(rows, crs=CRS_WGS84)
            
        except Exception as e:
            print(f"Shapely Voronoi failed: {e}. Using SciPy fallback.")
            result = _scipy_voronoi_fallback(pts_wgs, pts_proj, clip_union, bbox_proj)
    else:
        result = _scipy_voronoi_fallback(pts_wgs, pts_proj, clip_union, bbox_proj)
    
    # Apply distance cap: create buffer around each hospital and intersect
    geod = Geod(ellps="WGS84")
    capped_result = []
    for _, row in result.iterrows():
        hosp_name = row['Hospital']
        geom = row['geometry']
        
        if geom is None or geom.is_empty:
            capped_result.append({'Hospital': hosp_name, 'geometry': Polygon()})
            continue
        
        # Find hospital point
        hosp_row = pts_wgs[pts_wgs['Hospital'] == hosp_name]
        if hosp_row.empty:
            capped_result.append({'Hospital': hosp_name, 'geometry': Polygon()})
            continue
        
        hosp_point = hosp_row.geometry.iloc[0]
        hosp_lon = hosp_point.x
        hosp_lat = hosp_point.y
        
        # Create buffer polygon (5km radius)
        # Create a circle by creating points around the hospital
        angles = np.linspace(0, 360, 64)  # 64 points for smooth circle
        circle_points = []
        for angle in angles:
            lon2, lat2, _ = geod.fwd(hosp_lon, hosp_lat, angle, distance_cap_km * 1000)
            circle_points.append((lon2, lat2))
        circle_poly = Polygon(circle_points)
        
        # Intersect Voronoi region with circle
        capped_geom = geom.intersection(circle_poly)
        if capped_geom.is_empty:
            capped_geom = Polygon()
        elif isinstance(capped_geom, MultiPolygon):
            # Take the largest polygon if multiple
            capped_geom = max(capped_geom.geoms, key=lambda p: p.area)
        
        capped_result.append({'Hospital': hosp_name, 'geometry': capped_geom})
    
    return gpd.GeoDataFrame(capped_result, crs=CRS_WGS84)


def _scipy_voronoi_fallback(pts_wgs, pts_proj, clip_union, bbox_proj):
    """Fallback Voronoi using SciPy."""
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
    
    # Assign leftover area
    covered = unary_union([g for g in result.geometry if g is not None and not g.is_empty])
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


def calculate_catchment_area(gaza_gdf, hospitals_gdf, hospital_intervals, period_start, period_end):
    """
    Calculate catchment areas for each hospital during a time period using Voronoi polygons.
    Returns dict mapping hospital name to catchment area in km².
    Handles weighting if hospital status changes mid-period.
    """
    # Initialize geodetic calculator for accurate area calculation
    geod = Geod(ellps="WGS84")
    
    # Get all status change dates within the period
    change_dates = []
    for hosp_name in hospitals_gdf['Hospital'].values:
        if hosp_name in hospital_intervals:
            for dt, _ in hospital_intervals[hosp_name]:
                if period_start <= dt < period_end:
                    change_dates.append(dt)
    
    # Add period boundaries
    change_dates = sorted(set([period_start] + change_dates + [period_end]))
    
    # Calculate area for each sub-interval and weight by duration
    total_duration = (period_end - period_start).total_seconds()
    catchment_areas = {hosp: 0.0 for hosp in hospitals_gdf['Hospital'].values}
    
    for i in range(len(change_dates) - 1):
        sub_start = change_dates[i]
        sub_end = change_dates[i + 1]
        sub_duration = (sub_end - sub_start).total_seconds()
        weight = sub_duration / total_duration if total_duration > 0 else 0
        
        # Get open hospitals for this sub-interval (check at midpoint)
        # Only include target hospitals
        sub_midpoint = sub_start + (sub_end - sub_start) / 2
        open_hospitals_gdf = gpd.GeoDataFrame(columns=['Hospital', 'geometry'], crs=CRS_WGS84)
        
        for _, row in hospitals_gdf.iterrows():
            hosp_name = row['Hospital']
            matched_name = match_hospital_name(hosp_name)
            # Only process target hospitals
            if matched_name and matched_name in TARGET_HOSPITALS:
                status = get_hospital_status_at_date(hospital_intervals, hosp_name, sub_midpoint)
                if status == "Open":
                    # Get the geometry from hospitals_gdf
                    hosp_geom = hospitals_gdf[hospitals_gdf['Hospital'] == hosp_name].geometry.iloc[0]
                    open_hospitals_gdf = pd.concat([
                        open_hospitals_gdf,
                        gpd.GeoDataFrame([{'Hospital': hosp_name, 'geometry': hosp_geom}], crs=CRS_WGS84)
                    ], ignore_index=True)
        
        if open_hospitals_gdf.empty:
            continue
        
        # Create Voronoi polygons for open hospitals
        voronoi_polys = voronoi_polygons_clipped(open_hospitals_gdf, gaza_gdf, distance_cap_km=CATCHMENT_DISTANCE_KM)
        
        # Calculate area for each hospital using geodetic area calculation
        for _, row in voronoi_polys.iterrows():
            hosp_name = row['Hospital']
            geom = row['geometry']
            
            if geom is None or geom.is_empty:
                continue
            
            # Calculate area using geodetic calculation
            try:
                if isinstance(geom, MultiPolygon):
                    total_area_m2 = 0
                    for poly in geom.geoms:
                        area_m2, _ = geod.geometry_area_perimeter(poly)
                        total_area_m2 += abs(area_m2)
                else:
                    area_m2, _ = geod.geometry_area_perimeter(geom)
                    total_area_m2 = abs(area_m2)
                
                area_km2 = total_area_m2 / 1e6  # convert m² to km²
                # Weight by duration
                catchment_areas[hosp_name] += area_km2 * weight
            except Exception as e:
                print(f"Warning: Could not calculate area for {hosp_name}: {e}")
                continue
    
    return catchment_areas


def match_hospital_name(name):
    """
    Match hospital name to target hospitals.
    Returns the standardized name or None.
    """
    if not name or pd.isna(name):
        return None
    name_lower = str(name).lower().strip()
    for target, variations in HOSPITAL_NAME_MAPPING.items():
        for variant in variations:
            variant_lower = variant.lower()
            if variant_lower in name_lower or name_lower in variant_lower:
                return target
            # Also check for partial matches
            if "nasser" in name_lower and target == "Al Nasser":
                return target
            if ("european" in name_lower or "egh" in name_lower) and target == "EGH":
                return target
            if "shifa" in name_lower and target == "Al Shifa":
                return target
    return None


def create_html_visualization(gaza_gdf, hospitals_gdf, hospital_intervals, 
                              period_start, period_end, output_path):
    """
    Create HTML visualization of catchment areas for the first two-week period.
    """
    # Get open hospitals for visualization
    sub_midpoint = period_start + (period_end - period_start) / 2
    open_hospitals_gdf = gpd.GeoDataFrame(columns=['Hospital', 'geometry'], crs=CRS_WGS84)
    hosp_coords = {}
    
    for _, row in hospitals_gdf.iterrows():
        hosp_name = row['Hospital']
        matched_name = match_hospital_name(hosp_name)
        if matched_name and matched_name in TARGET_HOSPITALS:
            status = get_hospital_status_at_date(hospital_intervals, hosp_name, sub_midpoint)
            if status == "Open":
                hosp_geom = hospitals_gdf[hospitals_gdf['Hospital'] == hosp_name].geometry.iloc[0]
                open_hospitals_gdf = pd.concat([
                    open_hospitals_gdf,
                    gpd.GeoDataFrame([{'Hospital': hosp_name, 'geometry': hosp_geom}], crs=CRS_WGS84)
                ], ignore_index=True)
                hosp_coords[hosp_name] = (hosp_geom.y, hosp_geom.x)
    
    if open_hospitals_gdf.empty:
        print(f"No open target hospitals in period {period_start} to {period_end}")
        return
    
    # Create Voronoi polygons
    voronoi_polys = voronoi_polygons_clipped(open_hospitals_gdf, gaza_gdf, distance_cap_km=CATCHMENT_DISTANCE_KM)
    
    # Create map
    bounds = gaza_gdf.total_bounds
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    m = folium.Map(location=center, zoom_start=11)
    
    # Color mapping for hospitals
    colors = {
        "Al Nasser": "#e41a1c",
        "EGH": "#377eb8",
        "Al Shifa": "#4daf4a"
    }
    
    # Add catchment area polygons with borders
    for _, row in voronoi_polys.iterrows():
        hosp_name = row['Hospital']
        geom = row['geometry']
        matched_name = match_hospital_name(hosp_name)
        
        if geom is None or geom.is_empty:
            continue
        
        if matched_name:
            color = colors.get(matched_name, "#999999")
            # Add filled polygon with border
            folium.GeoJson(
                geom.__geo_interface__,
                style_function=lambda feature, fill_color=color: {
                    'fillColor': fill_color,
                    'fillOpacity': 0.5,
                    'color': 'black',
                    'weight': 2,
                    'opacity': 0.8
                },
                tooltip=hosp_name
            ).add_to(m)
    
    # Add hospital markers
    for hosp_name, (lat, lon) in hosp_coords.items():
        matched_name = match_hospital_name(hosp_name)
        if matched_name:
            folium.Marker(
                location=(lat, lon),
                popup=hosp_name,
                icon=folium.Icon(color='red', icon='hospital-o', prefix='fa')
            ).add_to(m)
    
    # Add Gaza boundary
    folium.GeoJson(
        gaza_gdf.__geo_interface__,
        style_function=lambda feature: {
            'fillColor': 'transparent',
            'color': 'black',
            'weight': 3,
            'opacity': 1.0
        }
    ).add_to(m)
    
    # Add title
    open_hosp_names = [h for h in open_hospitals_gdf['Hospital'].values]
    title_html = f"""
    <div style="position: fixed; top: 10px; left: 10px; z-index: 9999; 
                 background: white; padding: 8px; border: 1px solid grey; font-size:12px;">
        <b>Catchment Areas</b><br/>
        <b>Period:</b> {period_start.strftime('%Y-%m-%d')} → {period_end.strftime('%Y-%m-%d')}<br/>
        <b>Open hospitals:</b> {', '.join(open_hosp_names)}
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Add legend
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 10px; z-index: 9999; 
                 background: white; padding: 8px; border: 1px solid grey; font-size:12px;">
        <b>Hospital Catchment Areas</b><br/>
        <div style="margin-top: 6px;">
            <span style="background-color: #e41a1c; padding: 2px 8px; color: white; border: 1px solid black;">Al Nasser</span><br/>
            <span style="background-color: #377eb8; padding: 2px 8px; color: white; border: 1px solid black;">EGH</span><br/>
            <span style="background-color: #4daf4a; padding: 2px 8px; color: white; border: 1px solid black;">Al Shifa</span><br/>
        </div>
        <div style="margin-top: 6px; font-size: 10px;">
            Areas within 5km of closest open hospital
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    m.save(output_path)
    print(f"Saved HTML visualization to: {output_path}")


def main():
    print("=" * 80)
    print("Catchment Area Calculations")
    print("=" * 80)
    
    # Read hospitals
    print("Reading hospitals...")
    hospitals_df, schedule_meta = read_hospitals_table(HOSP_PATH)
    print(f"Loaded {len(hospitals_df)} hospitals")
    
    # Check which hospitals match target hospitals
    print("Available hospitals:", hospitals_df['Hospital'].tolist())
    matched_hospitals = []
    for hosp_name in hospitals_df['Hospital'].values:
        matched = match_hospital_name(hosp_name)
        if matched and matched in TARGET_HOSPITALS:
            matched_hospitals.append(hosp_name)
            print(f"  {hosp_name} -> {matched}")
    
    if not matched_hospitals:
        print("Warning: No target hospitals found. Using all hospitals...")
        matched_hospitals = hospitals_df['Hospital'].tolist()
    
    print(f"Target hospitals to analyze: {matched_hospitals}")
    
    # Build hospital intervals
    hospital_intervals = build_hospital_open_intervals(hospitals_df, schedule_meta)
    
    # Get date range from Excel (first and last date in schedule_meta)
    all_dates = [dt for _, _, dt in schedule_meta if dt is not None]
    
    # Also check dates in the actual data cells
    for _, row in hospitals_df.iterrows():
        for col, _, dt in schedule_meta:
            if col in row.index and pd.notnull(row.get(col)):
                # Try to parse the cell value as a date
                try:
                    cell_date = pd.to_datetime(row[col])
                    if pd.notnull(cell_date):
                        all_dates.append(cell_date.to_pydatetime())
                except:
                    pass
    
    if not all_dates:
        raise ValueError("Could not determine date range from Excel file. No dates found in schedule_meta.")
    
    global START_DATE, END_DATE
    START_DATE = min(all_dates)
    END_DATE = max(all_dates)
    print(f"Date range: {START_DATE.date()} to {END_DATE.date()}")
    
    # Create hospitals GeoDataFrame
    hospitals_gdf = gpd.GeoDataFrame(
        hospitals_df.dropna(subset=["lon", "lat"]),
        geometry=gpd.points_from_xy(
            hospitals_df["lon"].astype(float),
            hospitals_df["lat"].astype(float)
        ),
        crs=CRS_WGS84
    )
    
    # Load Gaza boundary
    print("Loading Gaza boundary...")
    gaza_gdf = gpd.read_file(GAZA_BOUNDARY)
    if gaza_gdf.crs is None:
        gaza_gdf = gaza_gdf.set_crs(CRS_WGS84)
    gaza_gdf = gaza_gdf.to_crs(CRS_WGS84)
    gaza_gdf = gaza_gdf.dissolve(by=None).reset_index(drop=True)
    
    # Calculate total Gaza area
    geod = Geod(ellps="WGS84")
    gaza_geom = gaza_gdf.unary_union
    if isinstance(gaza_geom, MultiPolygon):
        total_gaza_area_m2 = 0
        for poly in gaza_geom.geoms:
            area_m2, _ = geod.geometry_area_perimeter(poly)
            total_gaza_area_m2 += abs(area_m2)
    else:
        area_m2, _ = geod.geometry_area_perimeter(gaza_geom)
        total_gaza_area_m2 = abs(area_m2)
    total_gaza_area_km2 = total_gaza_area_m2 / 1e6
    print(f"Total Gaza area: {total_gaza_area_km2:.2f} km²")
    
    # Calculate catchment areas for two-week periods
    print("Calculating catchment areas for two-week periods...")
    results = []
    
    current_date = START_DATE
    period_num = 1
    
    while current_date < END_DATE:
        period_start = current_date
        period_end = min(current_date + timedelta(days=14), END_DATE)
        
        print(f"\nPeriod {period_num}: {period_start.date()} to {period_end.date()}")
        
        catchment_areas = calculate_catchment_area(
            gaza_gdf, hospitals_gdf, hospital_intervals, period_start, period_end
        )
        
        # Store results (only for target hospitals)
        row_data = {"Period": f"{period_start.date()} to {period_end.date()}"}
        for _, row in hospitals_gdf.iterrows():
            hosp_name = row['Hospital']
            matched_name = match_hospital_name(hosp_name)
            if matched_name and matched_name in TARGET_HOSPITALS:
                row_data[hosp_name] = catchment_areas.get(hosp_name, 0.0)
        
        results.append(row_data)
        
        # Create HTML for first period
        if period_num == 1:
            html_path = os.path.join(OUTPUT_DIR, f"catchment_visualization_{period_start.strftime('%Y%m%d')}_to_{period_end.strftime('%Y%m%d')}.html")
            create_html_visualization(
                gaza_gdf, hospitals_gdf, hospital_intervals,
                period_start, period_end, html_path
            )
        
        current_date = period_end
        period_num += 1
    
    # Create Excel output
    print("\nCreating Excel output...")
    excel_data = []
    
    # Get all unique hospital names from results that match target hospitals
    all_hospitals = set()
    for row in results:
        for k in row.keys():
            if k != "Period":
                matched_name = match_hospital_name(k)
                if matched_name and matched_name in TARGET_HOSPITALS:
                    all_hospitals.add(k)
    
    # Get period columns
    period_cols = [r["Period"] for r in results]
    
    # Create rows: one per hospital (area row + percentage row)
    for hosp_name in sorted(all_hospitals):
        matched_name = match_hospital_name(hosp_name)
        if matched_name and matched_name in TARGET_HOSPITALS:
            # Area row
            area_row = {"Hospital": hosp_name}
            area_values = []
            for result_row in results:
                period = result_row["Period"]
                area = result_row.get(hosp_name, 0.0)
                area_row[period] = area
                area_values.append(area)
            # Calculate average
            area_row["Average"] = np.mean(area_values) if area_values else 0.0
            excel_data.append(area_row)
            
            # Percentage row (empty hospital name, percentage values)
            pct_row = {"Hospital": ""}  # Empty name for percentage row
            for result_row in results:
                period = result_row["Period"]
                area = result_row.get(hosp_name, 0.0)
                pct = (area / total_gaza_area_km2 * 100) if total_gaza_area_km2 > 0 else 0.0
                pct_row[period] = f"{pct:.2f}%"
            # Calculate average percentage
            avg_pct = (np.mean(area_values) / total_gaza_area_km2 * 100) if total_gaza_area_km2 > 0 else 0.0
            pct_row["Average"] = f"{avg_pct:.2f}%"
            excel_data.append(pct_row)
    
    # Create DataFrame
    if excel_data:
        df = pd.DataFrame(excel_data)
        
        # Reorder columns: Hospital first, then periods, then Average
        cols = ["Hospital"] + period_cols + ["Average"]
        df = df[[c for c in cols if c in df.columns]]
        
        # Save Excel
        excel_path = os.path.join(OUTPUT_DIR, "catchment_areas_over_time.xlsx")
        df.to_excel(excel_path, index=False)
        print(f"Saved Excel file to: {excel_path}")
        print(f"\nCatchment areas (km²) and percentages:")
        print(df.to_string(index=False))
    else:
        print("Warning: No data to write to Excel file")
    
    print("\n" + "=" * 80)
    print("Calculation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
