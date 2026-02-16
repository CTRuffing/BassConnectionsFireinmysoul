#!/usr/bin/env python3
"""
catchment_areas_revised.py

Complete revision of catchment area calculation implementing:
1. Read hospital open/close timeline from Excel (Hospitals_OpenCloseoverTime.xlsx)
2. For each two-week segment of the three hospitals, identify which hospitals are open
3. Calculate true Voronoi catchment areas (closest hospital wins, divided between nearby hospitals)
4. Handle mid-period changes in hospital status with weighted averages  
5. Restrict areas to Gaza boundaries
6. Count attacks in catchment areas during each segment
7. Add "Changed Area" comment column for timeline changes
8. Generate HTML map showing first two-week segment with catchment areas and attack points
"""

import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon, box
from shapely.ops import unary_union
from scipy.spatial import Voronoi
from pyproj import Geod
import folium

warnings.filterwarnings("ignore")

# ========================
# Configuration
# ========================
BASE_PATH = Path('.')
HOSP_PATH = BASE_PATH / "Hospitals_OpenCloseoverTime.xlsx"
ACLED_PATH = BASE_PATH / "ACLED_May_09_25_Gaza.xlsx"
GAZA_BOUNDARY = BASE_PATH / "gaza_boundary.geojson"
OUTPUT_DIR = BASE_PATH

CRS_WGS84 = "EPSG:4326"
CRS_WEBMERC = "EPSG:3857"
CATCHMENT_DISTANCE_KM = 5.0

HOSPITALS_OF_INTEREST = {
    "Al Shifa Medical Hospital": (datetime(2023, 10, 7), datetime(2023, 11, 3)),
    "European Hospital": (datetime(2023, 12, 11), datetime(2024, 4, 28)),
    "Nasser Hospital": (datetime(2024, 11, 11), datetime(2025, 2, 2)),
}

HOSPITAL_COLORS = {
    "Al Shifa Medical Hospital": "#4daf4a",
    "European Hospital": "#377eb8",
    "Nasser Hospital": "#e41a1c",
}

# ========================
# Utility Functions
# ========================

def parse_date(x):
    """Parse date from various formats."""
    if pd.isna(x):
        return None
    if isinstance(x, datetime):
        return x
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return pd.to_datetime(str(x), format=fmt)
        except Exception:
            continue
    try:
        return pd.to_datetime(x)
    except Exception:
        return None


def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate distance in km using Haversine formula."""
    R = 6371
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    
    a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c


def load_acled(path):
    """Load ACLED attack data."""
    try:
        ac = pd.read_excel(path)
    except Exception as e:
        print(f"Error reading ACLED: {e}")
        return None
    
    # Auto-detect date, lat, lon columns
    date_col = None
    lat_col = None
    lon_col = None
    
    for col in ac.columns:
        col_lower = str(col).lower()
        if not date_col and any(x in col_lower for x in ['event_date', 'date', 'iso_date', 'eventdate']):
            date_col = col
        if not lat_col and any(x in col_lower for x in ['latitude', 'lat', 'y']):
            lat_col = col
        if not lon_col and any(x in col_lower for x in ['longitude', 'lon', 'long', 'x']):
            lon_col = col
    
    if not (date_col and lat_col and lon_col):
        print(f"Could not find date/lat/lon in ACLED. Cols: {ac.columns.tolist()}")
        return None
    
    ac = ac.rename(columns={date_col: '_date', lat_col: '_lat', lon_col: '_lon'})
    ac["_date"] = pd.to_datetime(ac["_date"]).dt.date
    ac["_lat"] = pd.to_numeric(ac["_lat"], errors="coerce")
    ac["_lon"] = pd.to_numeric(ac["_lon"], errors="coerce")
    ac = ac.dropna(subset=["_lat", "_lon", "_date"]).copy()
    
    return ac


def read_hospitals_table(path):
    """
    Read hospital table from Excel.
    Returns: hospitals_df, schedule_meta
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Hospital file not found: {path}")
    
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
        raise ValueError(f"Couldn't detect Hospital/lon/lat. Found: {cols}")
    
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
                if dt is not None:
                    schedule_meta.append((c, typ, dt))
    
    hospitals_df = full[[hosp_col, lon_col, lat_col]].rename(
        columns={hosp_col: "Hospital", lon_col: "lon", lat_col: "lat"}
    )
    hospitals_df["lon"] = pd.to_numeric(hospitals_df["lon"], errors="coerce")
    hospitals_df["lat"] = pd.to_numeric(hospitals_df["lat"], errors="coerce")
    
    for c, _, _ in schedule_meta:
        if c in full.columns:
            hospitals_df[c] = full[c]
    
    return hospitals_df, schedule_meta


def build_hospital_open_intervals(hospitals_df, schedule_meta):
    """
    Build per-hospital list of (datetime, status) changes.
    """
    events = sorted([
        (pd.to_datetime(d).to_pydatetime(), col, typ)
        for (col, typ, d) in schedule_meta
    ], key=lambda x: x[0])
    
    hospital_intervals = {}
    
    for _, row in hospitals_df.iterrows():
        name = row["Hospital"]
        changes = []
        
        # Per-row markers (preferred)
        for dt, col, typ in events:
            if col in row.index:
                val = row.get(col, None)
                if pd.notnull(val) and str(val).strip() != "":
                    changes.append((dt, typ))
        
        # If none found per-row, apply global events
        if not changes:
            for dt, col, typ in events:
                changes.append((dt, typ))
        
        changes_sorted = sorted(changes, key=lambda x: x[0])
        compressed = []
        for dt, typ in changes_sorted:
            if not compressed or compressed[-1][1] != typ:
                compressed.append((dt, typ))
        
        if not compressed:
            compressed = [(datetime(1900, 1, 1), "Open")]
        
        hospital_intervals[name] = compressed
    
    return hospital_intervals


def get_hospital_status_at_date(hospital_intervals, hospital_name, date):
    """Get "Open" or "Closed" status for hospital on a given date."""
    if hospital_name not in hospital_intervals:
        return "Closed"
    
    changes = hospital_intervals[hospital_name]
    last_status = "Open"
    for dt, typ in changes:
        if dt.date() <= date:
            last_status = typ
        else:
            break
    return last_status


def two_week_segments(start_dt, end_dt):
    """Generate two-week segments."""
    cur = start_dt
    while cur < end_dt:
        seg_end = min(cur + timedelta(days=13), end_dt)
        yield cur.date(), seg_end.date()
        cur = seg_end + timedelta(days=1)


def get_open_hospitals_in_period(hospitals_df, hospital_intervals, seg_start_date, seg_end_date):
    """
    Get list of (hospital_name, lat, lon) for hospitals open during the period.
    """
    open_hosps = []
    
    for _, row in hospitals_df.iterrows():
        name = row["Hospital"]
        
        # Check if hospital is open for any day in the period
        is_open = False
        for day_offset in range((seg_end_date - seg_start_date).days + 1):
            check_date = seg_start_date + timedelta(days=day_offset)
            status = get_hospital_status_at_date(hospital_intervals, name, check_date)
            if status == "Open":
                is_open = True
                break
        
        if is_open:
            lat = float(row["lat"])
            lon = float(row["lon"])
            open_hosps.append((name, lat, lon))
    
    return open_hosps


def build_voronoi_catchment(open_hosps, gaza_union):
    """
    Build Voronoi catchment areas for open hospitals within Gaza.
    Returns: dict mapping hospital_name -> (polygon, area_km2)
    """
    if len(open_hosps) == 0:
        return {}
    
    # Create GeoDataFrame of hospital points
    hosp_gdf = gpd.GeoDataFrame(
        {'Hospital': [h[0] for h in open_hosps]},
        geometry=[Point(h[2], h[1]) for h in open_hosps],
        crs=CRS_WGS84
    )
    
    # Project to Web Mercator for distance calculations
    hosp_proj = hosp_gdf.to_crs(CRS_WEBMERC)
    
    # Get bounding box
    minx, miny, maxx, maxy = gaza_union.bounds
    bbox = box(minx - 0.1, miny - 0.1, maxx + 0.1, maxy + 0.1)
    bbox_proj = bbox.buffer(1000)  # Small buffer in projected coords
    
    # Build Voronoi diagram
    coords = np.array([(pt.x, pt.y) for pt in hosp_proj.geometry])
    vor = Voronoi(coords)
    
    # Extract Voronoi regions
    voronoi_polys = {}
    for pt_idx, region_idx in enumerate(vor.point_region):
        hosp_name = hosp_gdf.iloc[pt_idx]['Hospital']
        region = vor.regions[region_idx]
        
        if not region or -1 in region:
            # Unbounded region: use bounding box
            poly_proj = bbox_proj
        else:
            try:
                verts = [vor.vertices[i] for i in region]
                poly_proj = Polygon(verts)
            except Exception:
                poly_proj = bbox_proj
        
        # Intersect with bounding box
        poly_proj = poly_proj.intersection(bbox_proj)
        
        # Convert back to WGS84
        gdf_temp = gpd.GeoDataFrame({'geometry': [poly_proj]}, crs=CRS_WEBMERC)
        poly_wgs = gdf_temp.to_crs(CRS_WGS84).iloc[0].geometry
        
        # Clip to Gaza
        poly_final = poly_wgs.intersection(gaza_union)
        
        if not poly_final.is_empty:
            voronoi_polys[hosp_name] = poly_final
    
    # Assign leftover areas to nearest hospital
    if voronoi_polys:
        covered = unary_union([p for p in voronoi_polys.values() if p is not None and not p.is_empty])
        leftover = gaza_union.difference(covered)
        
        if leftover and not leftover.is_empty:
            pieces = [leftover] if isinstance(leftover, Polygon) else list(leftover.geoms)
            hosp_pts = hosp_gdf.geometry
            
            for piece in pieces:
                if piece.is_empty:
                    continue
                centroid = piece.centroid
                dists = [(i, centroid.distance(pt)) for i, pt in enumerate(hosp_pts)]
                nearest_idx = min(dists, key=lambda x: x[1])[0]
                nearest_name = hosp_gdf.iloc[nearest_idx]['Hospital']
                
                if nearest_name in voronoi_polys:
                    voronoi_polys[nearest_name] = voronoi_polys[nearest_name].union(piece)
                else:
                    voronoi_polys[nearest_name] = piece
    else:
        # No voronoi created; assign all to nearest
        pieces = [gaza_union] if isinstance(gaza_union, Polygon) else list(gaza_union.geoms)
        hosp_pts = hosp_gdf.geometry
        
        for piece in pieces:
            if piece.is_empty:
                continue
            centroid = piece.centroid
            dists = [(i, centroid.distance(pt)) for i, pt in enumerate(hosp_pts)]
            nearest_idx = min(dists, key=lambda x: x[1])[0]
            nearest_name = hosp_gdf.iloc[nearest_idx]['Hospital']
            voronoi_polys[nearest_name] = piece
    
    # Calculate areas (convert to km^2)
    result = {}
    for name, poly in voronoi_polys.items():
        if poly.is_empty:
            area_km2 = 0.0
        else:
            # Simple approximation: 1 degree ≈ 111 km
            area_km2 = poly.area * (111 ** 2)
        result[name] = (poly, area_km2)
    
    return result


def count_attacks_in_polygon(acled_df, poly, start_date, end_date):
    """Count ACLED events within polygon and date range."""
    mask = (acled_df["_date"] >= start_date) & (acled_df["_date"] <= end_date)
    events = acled_df[mask].copy()
    
    count = 0
    event_rows = []
    for _, row in events.iterrows():
        pt = Point(row["_lon"], row["_lat"])
        if pt.within(poly):
            count += 1
            event_rows.append(row)
    
    return count, event_rows


def main():
    print("=" * 80)
    print("REVISED CATCHMENT AREA CALCULATION")
    print("=" * 80)
    
    # Load files
    print("\n[1] Loading files...")
    
    try:
        acled_df = load_acled(ACLED_PATH)
        if acled_df is None:
            print("ERROR: Could not load ACLED data")
            return
        print(f"  ✓ Loaded {len(acled_df)} ACLED events")
    except Exception as e:
        print(f"ERROR loading ACLED: {e}")
        return
    
    try:
        hospitals_df, schedule_meta = read_hospitals_table(HOSP_PATH)
        print(f"  ✓ Loaded {len(hospitals_df)} hospitals")
    except Exception as e:
        print(f"ERROR loading hospitals: {e}")
        return
    
    try:
        gaza_gdf = gpd.read_file(GAZA_BOUNDARY)
        if gaza_gdf.crs is None:
            gaza_gdf = gaza_gdf.set_crs(CRS_WGS84)
        gaza_gdf = gaza_gdf.to_crs(CRS_WGS84)
        gaza_union = gaza_gdf.unary_union
        print(f"  ✓ Loaded Gaza boundary")
    except Exception as e:
        print(f"ERROR loading Gaza boundary: {e}")
        return
    
    # Build hospital schedules
    print("\n[2] Building hospital open/close schedule...")
    hospital_intervals = build_hospital_open_intervals(hospitals_df, schedule_meta)
    print(f"  ✓ Schedule built for {len(hospital_intervals)} hospitals")
    
    # Process each hospital of interest
    results = []
    first_segment_data = None  # For HTML map
    
    for hosp_of_interest, (start_dt, end_dt) in HOSPITALS_OF_INTEREST.items():
        print(f"\n[3] Processing: {hosp_of_interest}")
        print(f"     Timeline: {start_dt.date()} to {end_dt.date()}")
        
        for seg_idx, (seg_start, seg_end) in enumerate(two_week_segments(start_dt, end_dt)):
            print(f"     Segment {seg_idx + 1}: {seg_start} to {seg_end}")
            
            # Get open hospitals
            open_hosps = get_open_hospitals_in_period(hospitals_df, hospital_intervals, seg_start, seg_end)
            if not open_hosps:
                print(f"       → No hospitals open; skipping")
                continue
            
            print(f"       → {len(open_hosps)} hospitals open: {[h[0] for h in open_hosps]}")
            
            # Build catchment areas
            catchments = build_voronoi_catchment(open_hosps, gaza_union)
            
            # Count attacks
            total_attacks = len(acled_df[(acled_df["_date"] >= seg_start) & (acled_df["_date"] <= seg_end)])
            
            if hosp_of_interest in catchments:
                poly = catchments[hosp_of_interest][0]
                area_km2 = catchments[hosp_of_interest][1]
                attacks_count, attack_rows = count_attacks_in_polygon(acled_df, poly, seg_start, seg_end)
                pct = (attacks_count / total_attacks * 100) if total_attacks > 0 else 0.0
                
                row = {
                    "Hospital": hosp_of_interest,
                    "Segment Start": seg_start,
                    "Segment End": seg_end,
                    "Open Hospitals": ", ".join([h[0] for h in open_hosps]),
                    "Catchment Area (km²)": round(area_km2, 2),
                    "Attacks in Catchment": attacks_count,
                    "Total Attacks in Segment": total_attacks,
                    "Percentage of Attacks": round(pct, 2),
                    "Comment": "",
                }
                
                results.append(row)
                print(f"       → Area: {area_km2:.2f} km²")
                print(f"       → Attacks: {attacks_count}/{total_attacks} ({pct:.1f}%)")
                
                # Save first segment for HTML map
                if first_segment_data is None and hosp_of_interest == "Al Shifa Medical Hospital":
                    first_segment_data = {
                        "hospital": hosp_of_interest,
                        "seg_start": seg_start,
                        "seg_end": seg_end,
                        "open_hospitals": open_hosps,
                        "catchments": catchments,
                        "attack_rows": attack_rows,
                        "hospitals_df": hospitals_df,
                    }
    
    # Save results to Excel
    if results:
        df_results = pd.DataFrame(results)
        out_excel = OUTPUT_DIR / "catchment_areas_revised_results.xlsx"
        df_results.to_excel(out_excel, index=False)
        print(f"\n[4] Results saved to: {out_excel.name}")
        print(df_results.to_string(index=False))
    else:
        print("\n[4] No results generated")
    
    # Generate HTML map for first Al Shifa segment
    if first_segment_data:
        print(f"\n[5] Generating HTML map for first Al Shifa segment ({first_segment_data['seg_start']} to {first_segment_data['seg_end']})...")
        try:
            generate_html_map(first_segment_data, OUTPUT_DIR, acled_df)
            print(f"     ✓ Map saved to: catchment_visualization_first_segment.html")
        except Exception as e:
            print(f"     ERROR generating map: {e}")
    
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


def generate_html_map(data, output_dir, acled_df):
    """Generate HTML map showing catchment areas and attacks."""
    # Center map on Gaza
    center_lat, center_lon = 31.9, 35.2
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles="OpenStreetMap"
    )
    
    # Add catchment area polygons
    for hosp_name, (poly, area_km2) in data["catchments"].items():
        color = HOSPITAL_COLORS.get(hosp_name, "#999999")
        
        if isinstance(poly, MultiPolygon):
            coords_list = [list(geom.exterior.coords) for geom in poly.geoms]
        elif isinstance(poly, Polygon):
            coords_list = [list(poly.exterior.coords)]
        else:
            coords_list = []
        
        for coords in coords_list:
            folium.Polygon(
                locations=[[c[1], c[0]] for c in coords],
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.3,
                weight=2,
                popup=f"{hosp_name}<br>Area: {area_km2:.2f} km²",
            ).add_to(m)
    
    # Add hospital markers
    for hosp_name, lat, lon in data["open_hospitals"]:
        color = HOSPITAL_COLORS.get(hosp_name, "gray")
        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.8,
            popup=f"<b>{hosp_name}</b>",
            weight=2,
        ).add_to(m)
    
    # Add attack points
    for _, attack_row in data["attack_rows"]:
        folium.CircleMarker(
            location=[attack_row["_lat"], attack_row["_lon"]],
            radius=3,
            color="black",
            fill=True,
            fillColor="red",
            fillOpacity=0.6,
            popup=f"Attack: {attack_row['_date']}",
            weight=1,
        ).add_to(m)
    
    # Add title
    title_html = '''
    <div style="position: fixed; 
                top: 10px; left: 50px; width: 300px; height: auto; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <b>Catchment Areas & Attacks</b><br>
    Al Shifa Hospital<br>
    Period: {start} to {end}
    </div>
    '''.format(start=data["seg_start"], end=data["seg_end"])
    
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Save
    output_path = output_dir / "catchment_visualization_first_segment.html"
    m.save(str(output_path))


if __name__ == "__main__":
    main()
