#!/usr/bin/env python3
"""
test_calculatecatchmentarea.py

Implements the full catchment area pipeline per the specification:

1. Availability timeline: Parse hospital open/close dates from Hospitals_OpenCloseoverTime.xlsx
2. Data timeline: Analyze three hospital periods (Al Shifa, European Hospital, Nasser Hospital)
3. Data aggregation: Split each data timeline into two-week segments
4. Catchment area method: Voronoi-based areas within 5 km, clipped to Gaza (gaza_boundary.geojson)
5. Weighted catchment area method: When hospital list changes mid-segment, weight by sub-segment days
6. Attack count method: Count ACLED attacks within each catchment during each segment
7. Total counts method: Total attacks per segment (regardless of location)
8. Output method: Excel with segment details, catchment areas, attack counts, percentages
9. Output Test HTML Map: First two-week segment of Nasser Hospital with markers and polygons

Hospital spreadsheet format (Hospitals_OpenCloseoverTime.xlsx):
- Column A: Hospital name
- Columns B, C: Longitude, Latitude
- Columns D onwards: Alternate Open, Closed, Open, Closed... dates (per hospital row)
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon, MultiPoint, box
from shapely.ops import unary_union
from pyproj import Geod

# Optional: shapely voronoi (faster when available)
try:
    from shapely.ops import voronoi_diagram
    from shapely import geometry as shapely_geom
    _HAS_SHAPELY_VORONOI = True
except ImportError:
    voronoi_diagram = None
    shapely_geom = None
    _HAS_SHAPELY_VORONOI = False

from scipy.spatial import Voronoi
import folium

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_PATH = Path(__file__).parent.resolve()
HOSP_PATH = BASE_PATH / "Hospitals_OpenCloseoverTime.xlsx"
ACLED_PATH = BASE_PATH / "ACLED_May_09_25_Gaza.xlsx"
GAZA_GEOJSON = BASE_PATH / "gaza_boundary.geojson"
OUTPUT_DIR = BASE_PATH

CRS_WGS84 = "EPSG:4326"
CRS_WEBMERC = "EPSG:3857"
CATCHMENT_DISTANCE_KM = 5.0
VORONOI_BUFFER_METERS = 15000

# Data timelines of interest: (hospital_name_substring, start_date, end_date)
DATA_TIMELINES = [
    ("Al Shifa Medical Hospital", datetime(2023, 10, 7), datetime(2023, 11, 3)),
    ("European Hospital", datetime(2023, 12, 11), datetime(2024, 4, 28)),
    ("Nasser Hospital", datetime(2024, 11, 11), datetime(2025, 2, 2)),
]

# ---------------------------------------------------------------------------
# Availability Timeline
# ---------------------------------------------------------------------------

def read_hospitals_open_close(path: Path) -> Tuple[pd.DataFrame, Dict[str, List[Tuple[datetime, str]]]]:
    """
    Read hospital open/close dates from Hospitals_OpenCloseoverTime.xlsx.
    
    Spreadsheet format: Col A=Hospital name, B=Longitude, C=Latitude,
    then columns D onwards alternate Open, Closed, Open, Closed... dates.
    
    Two layouts supported:
    (1) Dates in header row (row 1): Row 0 has "Open"/"Closed" labels, row 1 has dates.
        All hospitals share the same schedule. Row 2+ = hospital data.
    (2) Dates per row: Each hospital row has its own dates in columns D+.
    
    Returns:
        hospitals_df: DataFrame with columns Hospital, lon, lat
        availability_timeline: Dict mapping hospital_name -> [(datetime, 'Open'|'Closed'), ...]
    """
    if not path.exists():
        raise FileNotFoundError(f"Hospital file not found: {path}")
    
    raw = pd.read_excel(path, header=None)
    
    hosp_col, lon_col, lat_col = 0, 1, 2
    schedule_cols = list(range(3, raw.shape[1]))
    
    # Detect layout: if row 1 has date-like values in col 3, use dates-in-header layout
    dates_in_header = False
    if len(raw) >= 2 and schedule_cols:
        header_date = _parse_date(raw.iloc[1, 3])
        if header_date is not None:
            dates_in_header = True
    
    schedule_events: List[Tuple[datetime, str]] = []
    if dates_in_header:
        for i, col_idx in enumerate(schedule_cols):
            if col_idx >= raw.shape[1]:
                break
            status = "Open" if i % 2 == 0 else "Closed"
            dt = _parse_date(raw.iloc[1, col_idx])
            if dt is not None:
                schedule_events.append((dt, status))
        schedule_events = sorted(schedule_events, key=lambda x: x[0])
        compressed = []
        for dt, typ in schedule_events:
            if not compressed or compressed[-1][1] != typ:
                compressed.append((dt, typ))
        if not compressed:
            compressed = [(datetime(1900, 1, 1), "Open")]
        schedule_events = compressed
        data_start_row = 2
    else:
        data_start_row = 1
    
    hospitals_rows: List[Dict] = []
    availability_timeline: Dict[str, List[Tuple[datetime, str]]] = {}
    
    for raw_row_idx in range(data_start_row, len(raw)):
        name_val = raw.iloc[raw_row_idx, hosp_col]
        lon_val = pd.to_numeric(raw.iloc[raw_row_idx, lon_col], errors="coerce")
        lat_val = pd.to_numeric(raw.iloc[raw_row_idx, lat_col], errors="coerce")
        if pd.isna(lon_val) or pd.isna(lat_val):
            continue
        name = str(name_val).strip()
        if not name or name == "nan":
            continue
        
        hospitals_rows.append({"Hospital": name, "lon": float(lon_val), "lat": float(lat_val)})
        
        if dates_in_header:
            availability_timeline[name] = schedule_events
        else:
            events: List[Tuple[datetime, str]] = []
            for i, col_idx in enumerate(schedule_cols):
                if col_idx >= raw.shape[1]:
                    break
                status = "Open" if i % 2 == 0 else "Closed"
                val = raw.iloc[raw_row_idx, col_idx]
                dt = _parse_date(val)
                if dt is not None:
                    events.append((dt, status))
            events = sorted(events, key=lambda x: x[0])
            compressed_list: List[Tuple[datetime, str]] = []
            for dt, typ in events:
                if not compressed_list or compressed_list[-1][1] != typ:
                    compressed_list.append((dt, typ))
            if not compressed_list:
                compressed_list = [(datetime(1900, 1, 1), "Open")]
            availability_timeline[name] = compressed_list
    
    hospitals_df = pd.DataFrame(hospitals_rows)
    return hospitals_df, availability_timeline


def _parse_date(x: Any) -> Optional[datetime]:
    """Parse date from Excel cell value."""
    if pd.isna(x) or x is None or str(x).strip() == "":
        return None
    if isinstance(x, datetime):
        return x
    try:
        return pd.to_datetime(x).to_pydatetime()
    except Exception:
        return None


def get_hospital_status_at_date(availability_timeline: Dict[str, List[Tuple[datetime, str]]],
                                hospital_name: str, date: datetime) -> str:
    """
    Get 'Open' or 'Closed' status for a hospital on a given date.
    """
    if hospital_name not in availability_timeline:
        return "Closed"
    changes = availability_timeline[hospital_name]
    last_status = "Open"
    for dt, typ in changes:
        if dt <= date:
            last_status = typ
        else:
            break
    return last_status


# ---------------------------------------------------------------------------
# Data Timeline & Aggregation
# ---------------------------------------------------------------------------

def data_aggregation_method(start_date: datetime, end_date: datetime) -> List[Tuple[datetime, datetime]]:
    """
    Split data timeline into two-week segments.
    Last segment may be less than 14 days.
    
    Returns list of (segment_start, segment_end) inclusive.
    """
    segments: List[Tuple[datetime, datetime]] = []
    cur = start_date
    while cur <= end_date:
        seg_end = min(cur + timedelta(days=13), end_date)
        segments.append((cur, seg_end))
        cur = seg_end + timedelta(days=1)
    return segments


def get_open_hospitals_during_segment(hospitals_df: pd.DataFrame,
                                     availability_timeline: Dict[str, List[Tuple[datetime, str]]],
                                     seg_start: datetime, seg_end: datetime) -> List[Tuple[str, float, float]]:
    """
    Determine which hospitals were open during the two-week segment.
    Returns list of (hospital_name, lat, lon).
    """
    open_hospitals: List[Tuple[str, float, float]] = []
    for _, row in hospitals_df.iterrows():
        name = str(row["Hospital"]).strip()
        for day_offset in range((seg_end.date() - seg_start.date()).days + 1):
            check_date = seg_start + timedelta(days=day_offset)
            if get_hospital_status_at_date(availability_timeline, name, check_date) == "Open":
                open_hospitals.append((name, float(row["lat"]), float(row["lon"])))
                break
    return open_hospitals


# ---------------------------------------------------------------------------
# Catchment Area Method
# ---------------------------------------------------------------------------

def load_gaza_boundary(path: Path) -> Any:
    """Load Gaza boundary from GeoJSON file."""
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(CRS_WGS84)
    gdf = gdf.to_crs(CRS_WGS84)
    return gdf.unary_union


def catchment_area_method(open_hospitals: List[Tuple[str, float, float]],
                         gaza_union: Any,
                         distance_cap_km: float = CATCHMENT_DISTANCE_KM) -> Dict[str, Tuple[Any, float]]:
    """
    Determine catchment areas for hospitals open during the segment.
    
    Catchment = Voronoi region (closest hospital) clipped to Gaza boundary,
    intersected with 5 km geodesic buffer around each hospital.
    
    Returns: Dict mapping hospital_name -> (polygon, area_km2)
    """
    if not open_hospitals:
        return {}
    
    hosp_gdf = gpd.GeoDataFrame(
        {"Hospital": [h[0] for h in open_hospitals]},
        geometry=[Point(h[2], h[1]) for h in open_hospitals],
        crs=CRS_WGS84
    )
    hosp_proj = hosp_gdf.to_crs(CRS_WEBMERC)
    
    gaza_proj = gpd.GeoSeries([gaza_union], crs=CRS_WGS84).to_crs(CRS_WEBMERC)
    minx, miny, maxx, maxy = gaza_proj.total_bounds
    bbox_proj = box(minx - VORONOI_BUFFER_METERS, miny - VORONOI_BUFFER_METERS,
                   maxx + VORONOI_BUFFER_METERS, maxy + VORONOI_BUFFER_METERS)
    
    if _HAS_SHAPELY_VORONOI:
        try:
            multip = shapely_geom.MultiPoint([(pt.x, pt.y) for pt in hosp_proj.geometry])
            vor = voronoi_diagram(multip, envelope=bbox_proj, tolerance=0.0)
            polys_proj = _extract_voronoi_polygons(vor, bbox_proj, hosp_proj, hosp_gdf)
        except Exception:
            polys_proj = _scipy_voronoi_clipped(hosp_proj, hosp_gdf, bbox_proj, gaza_union)
    else:
        polys_proj = _scipy_voronoi_clipped(hosp_proj, hosp_gdf, bbox_proj, gaza_union)
    
    # Clip to Gaza and apply 5 km distance cap
    geod = Geod(ellps="WGS84")
    result: Dict[str, Tuple[Any, float]] = {}
    
    for hosp_name, poly in polys_proj.items():
        if poly is None or poly.is_empty:
            result[hosp_name] = (Polygon(), 0.0)
            continue
        poly_clipped = poly.intersection(gaza_union)
        if poly_clipped.is_empty:
            result[hosp_name] = (Polygon(), 0.0)
            continue
        
        # 5 km geodesic buffer around this hospital
        hosp_row = hosp_gdf[hosp_gdf["Hospital"] == hosp_name]
        if hosp_row.empty:
            result[hosp_name] = (poly_clipped, 0.0)
            continue
        pt = hosp_row.geometry.iloc[0]
        angles = np.linspace(0, 360, 128)
        circle_pts = []
        for a in angles:
            lon2, lat2, _ = geod.fwd(pt.x, pt.y, a, distance_cap_km * 1000)
            circle_pts.append((lon2, lat2))
        circle_poly = Polygon(circle_pts)
        capped = poly_clipped.intersection(circle_poly)
        if capped.is_empty:
            result[hosp_name] = (Polygon(), 0.0)
            continue
        if isinstance(capped, MultiPolygon):
            capped = max(capped.geoms, key=lambda p: p.area)
        area_m2, _ = geod.geometry_area_perimeter(capped)
        area_km2 = abs(area_m2) / 1e6
        result[hosp_name] = (capped, area_km2)
    
    return result


def _extract_voronoi_polygons(vor, bbox_proj, hosp_proj, hosp_gdf) -> Dict[str, Any]:
    """Extract polygons from shapely voronoi_diagram and assign to hospitals."""
    poly_list = []
    if hasattr(vor, "geoms"):
        for g in vor.geoms:
            if isinstance(g, (Polygon, MultiPolygon)):
                poly_list.append(g)
    elif isinstance(vor, (Polygon, MultiPolygon)):
        poly_list = [vor]
    
    if not poly_list:
        return {}
    
    polys_wgs = gpd.GeoDataFrame(geometry=poly_list, crs=CRS_WEBMERC).to_crs(CRS_WGS84)
    assigned: List[Tuple[str, Any]] = []
    hosp_pts = hosp_gdf.geometry
    
    for poly in polys_wgs.geometry:
        if poly is None or poly.is_empty:
            continue
        rep = poly.representative_point()
        dists = hosp_pts.distance(rep)
        nearest_idx = int(dists.idxmin())
        hosp_name = hosp_gdf.iloc[nearest_idx]["Hospital"]
        assigned.append((hosp_name, poly))
    
    result: Dict[str, Any] = {}
    for hosp in hosp_gdf["Hospital"].values:
        polys_for = [p for (h, p) in assigned if h == hosp]
        result[hosp] = unary_union(polys_for) if polys_for else Polygon()
    return result


def _scipy_voronoi_clipped(hosp_proj, hosp_gdf, bbox_proj, gaza_union) -> Dict[str, Any]:
    """Fallback: SciPy Voronoi with bounding box."""
    coords = np.array([(pt.x, pt.y) for pt in hosp_proj.geometry])
    vor = Voronoi(coords)
    hosp_names = hosp_gdf["Hospital"].values
    
    def make_bounded_poly(site: np.ndarray) -> Polygon:
        bbox_coords = np.array(bbox_proj.exterior.coords)
        pts = np.vstack([site, bbox_coords])
        multipoint = MultiPoint([tuple(p) for p in pts])
        return multipoint.convex_hull.intersection(bbox_proj)
    
    proj_polys = []
    for pt_idx, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if not region or -1 in region:
            poly = make_bounded_poly(coords[pt_idx])
        else:
            try:
                verts = [vor.vertices[i] for i in region]
                poly = Polygon(verts).intersection(bbox_proj)
            except Exception:
                poly = make_bounded_poly(coords[pt_idx])
        proj_polys.append(poly)
    
    polys_wgs = gpd.GeoDataFrame(geometry=proj_polys, crs=CRS_WEBMERC).to_crs(CRS_WGS84)
    result = {}
    for i, hosp_name in enumerate(hosp_names):
        try:
            clipped = polys_wgs.iloc[i].geometry.intersection(gaza_union)
        except Exception:
            clipped = Polygon()
        result[hosp_name] = clipped if clipped and not clipped.is_empty else Polygon()
    
    # Assign leftover to nearest
    covered = unary_union([g for g in result.values() if g and not g.is_empty])
    leftover = gaza_union.difference(covered) if covered else gaza_union
    if leftover and not leftover.is_empty:
        pieces = [leftover] if isinstance(leftover, Polygon) else list(leftover.geoms)
        hosp_pts = hosp_gdf.geometry
        for piece in pieces:
            if piece.is_empty:
                continue
            rep = piece.representative_point()
            dists = hosp_pts.distance(rep)
            nearest_idx = int(dists.idxmin())
            n = hosp_names[nearest_idx]
            cur = result.get(n, Polygon())
            result[n] = cur.union(piece) if cur and not cur.is_empty else piece
    
    return result


# ---------------------------------------------------------------------------
# Weighted Catchment Area Method
# ---------------------------------------------------------------------------

def get_change_dates_in_segment(availability_timeline: Dict[str, List[Tuple[datetime, str]]],
                                open_hospital_names: List[str],
                                seg_start: datetime, seg_end: datetime) -> List[datetime]:
    """Get dates when hospital status changes within the segment."""
    change_dates = {seg_start, seg_end}
    for name in open_hospital_names:
        if name not in availability_timeline:
            continue
        for dt, _ in availability_timeline[name]:
            if seg_start < dt < seg_end:
                change_dates.add(dt)
    return sorted(change_dates)


def weighted_catchment_area_method(open_hospitals: List[Tuple[str, float, float]],
                                  hospitals_df: pd.DataFrame,
                                  availability_timeline: Dict[str, List[Tuple[datetime, str]]],
                                  gaza_union: Any,
                                  seg_start: datetime, seg_end: datetime) -> Dict[str, float]:
    """
    If hospital list changes during segment, split into sub-segments, compute
    catchment for each, weight by days, combine. Otherwise return normal catchment areas.
    """
    open_names = [h[0] for h in open_hospitals]
    change_dates = get_change_dates_in_segment(
        availability_timeline, open_names, seg_start, seg_end
    )
    total_days = (seg_end - seg_start).days + 1
    
    weighted_areas: Dict[str, float] = {}
    
    for i in range(len(change_dates) - 1):
        sub_start = change_dates[i]
        sub_end = change_dates[i + 1]
        sub_days = (sub_end - sub_start).days
        if sub_days <= 0:
            continue
        weight = sub_days / total_days
        
        sub_open = get_open_hospitals_during_segment(
            hospitals_df, availability_timeline, sub_start, sub_end - timedelta(seconds=1)
        )
        if not sub_open:
            continue
        
        catchments = catchment_area_method(sub_open, gaza_union)
        for hosp_name, (_, area_km2) in catchments.items():
            weighted_areas[hosp_name] = weighted_areas.get(hosp_name, 0.0) + area_km2 * weight
    
    if not weighted_areas:
        catchments = catchment_area_method(open_hospitals, gaza_union)
        weighted_areas = {h: a for h, (_, a) in catchments.items()}
    
    return weighted_areas


# ---------------------------------------------------------------------------
# Attack Count & Total Counts Methods
# ---------------------------------------------------------------------------

def load_acled(path: Path) -> pd.DataFrame:
    """Load ACLED attack data, auto-detect date/lat/lon columns."""
    df = pd.read_excel(path)
    date_col = None
    lat_col = None
    lon_col = None
    for c in df.columns:
        cl = str(c).lower()
        if not date_col and ("event_date" in cl or "date" in cl or "eventdate" in cl):
            date_col = c
        if not lat_col and ("lat" in cl or "latitude" in cl or cl == "y"):
            lat_col = c
        if not lon_col and ("lon" in cl or "longitude" in cl or cl == "x"):
            lon_col = c
    
    if not (date_col and lat_col and lon_col):
        raise ValueError(f"Cannot find date/lat/lon in ACLED. Columns: {list(df.columns)}")
    
    df = df.rename(columns={date_col: "_date", lat_col: "_lat", lon_col: "_lon"})
    df["_date"] = pd.to_datetime(df["_date"]).dt.date
    df["_lat"] = pd.to_numeric(df["_lat"], errors="coerce")
    df["_lon"] = pd.to_numeric(df["_lon"], errors="coerce")
    return df.dropna(subset=["_date", "_lat", "_lon"]).copy()


def attack_count_method(acled_df: pd.DataFrame, polygon: Any,
                       seg_start: datetime, seg_end: datetime) -> int:
    """Count attacks within catchment polygon during segment."""
    mask = (acled_df["_date"] >= seg_start.date()) & (acled_df["_date"] <= seg_end.date())
    subset = acled_df[mask]
    count = 0
    for _, row in subset.iterrows():
        pt = Point(row["_lon"], row["_lat"])
        if pt.within(polygon):
            count += 1
    return count


def total_counts_method(acled_df: pd.DataFrame, seg_start: datetime, seg_end: datetime) -> int:
    """Count total attacks during segment (regardless of location)."""
    mask = (acled_df["_date"] >= seg_start.date()) & (acled_df["_date"] <= seg_end.date())
    return int(mask.sum())


# ---------------------------------------------------------------------------
# Output Method
# ---------------------------------------------------------------------------

def output_method(results: List[Dict], output_path: Path) -> None:
    """
    Write results to Excel.
    Each row: Data timeline segment, Hospital name, Catchment area (km²),
    Attacks in catchment, Percentage, Total attacks.
    """
    df = pd.DataFrame(results)
    df.to_excel(output_path, index=False)


# ---------------------------------------------------------------------------
# Output Test HTML Map
# ---------------------------------------------------------------------------

def output_test_html_map(open_hospitals: List[Tuple[str, float, float]],
                        catchments: Dict[str, Tuple[Any, float]],
                        seg_start: datetime, seg_end: datetime,
                        output_path: Path) -> None:
    """
    Create HTML map for first two-week segment of Nasser Hospital:
    - Hospital markers with names
    - Catchment area polygons
    """
    center_lat = sum(h[1] for h in open_hospitals) / len(open_hospitals)
    center_lon = sum(h[2] for h in open_hospitals) / len(open_hospitals)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="OpenStreetMap")
    
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
    for i, (hosp_name, lat, lon) in enumerate(open_hospitals):
        color = colors[i % len(colors)]
        folium.CircleMarker(
            location=[lat, lon], radius=10, color=color, fill=True, fillColor=color,
            fillOpacity=0.9, popup=f"<b>{hosp_name}</b>", weight=2
        ).add_to(m)
    
    for i, (hosp_name, (poly, area_km2)) in enumerate(catchments.items()):
        if poly is None or poly.is_empty:
            continue
        color = colors[i % len(colors)]
        if isinstance(poly, MultiPolygon):
            for geom in poly.geoms:
                _add_polygon_to_map(m, geom, color, f"{hosp_name} ({area_km2:.2f} km²)")
        else:
            _add_polygon_to_map(m, poly, color, f"{hosp_name} ({area_km2:.2f} km²)")
    
    title_html = f"""
    <div style="position: fixed; top: 10px; left: 50px; width: 350px; padding: 10px;
                background: white; border: 2px solid grey; z-index: 9999; font-size: 14px;">
    <b>Catchment Areas & Hospital Locations</b><br>
    Period: {seg_start.strftime('%Y-%m-%d')} to {seg_end.strftime('%Y-%m-%d')}
    </div>"""
    m.get_root().html.add_child(folium.Element(title_html))
    m.save(str(output_path))


def _add_polygon_to_map(m: folium.Map, poly: Polygon, color: str, tooltip: str) -> None:
    """Add a polygon to folium map."""
    if poly.is_empty or not poly.exterior:
        return
    coords = [[c[1], c[0]] for c in poly.exterior.coords]
    folium.Polygon(
        locations=coords, color=color, fill=True, fillColor=color,
        fillOpacity=0.3, weight=2, tooltip=tooltip
    ).add_to(m)


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full catchment area pipeline."""
    print("=" * 70)
    print("Catchment Area Calculation Pipeline")
    print("=" * 70)
    
    # 1. Load availability timeline
    print("\n[1] Reading hospital open/close dates (availability timeline)...")
    hospitals_df, availability_timeline = read_hospitals_open_close(HOSP_PATH)
    print(f"    Loaded {len(hospitals_df)} hospitals")
    
    # 2. Load Gaza boundary
    print("\n[2] Loading Gaza boundary...")
    gaza_union = load_gaza_boundary(GAZA_GEOJSON)
    
    # 3. Load ACLED
    print("\n[3] Loading ACLED attack data...")
    acled_df = load_acled(ACLED_PATH)
    print(f"    Loaded {len(acled_df)} attack events")
    
    # 4. Process each data timeline
    results: List[Dict] = []
    first_nasser_segment_data: Optional[Dict] = None
    
    for hosp_substring, start_dt, end_dt in DATA_TIMELINES:
        segments = data_aggregation_method(start_dt, end_dt)
        print(f"\n[4] Processing {hosp_substring}: {len(segments)} segments")
        
        for seg_start, seg_end in segments:
            open_hospitals = get_open_hospitals_during_segment(
                hospitals_df, availability_timeline, seg_start, seg_end
            )
            if not open_hospitals:
                continue
            
            # Catchment areas (weighted if hospital list changes)
            weighted_areas = weighted_catchment_area_method(
                open_hospitals, hospitals_df, availability_timeline, gaza_union, seg_start, seg_end
            )
            catchments = catchment_area_method(open_hospitals, gaza_union)
            
            total_attacks = total_counts_method(acled_df, seg_start, seg_end)
            
            for hosp_name in open_hospitals:
                h_name = hosp_name[0]
                area_km2 = weighted_areas.get(h_name, catchments.get(h_name, (None, 0.0))[1])
                poly = catchments.get(h_name, (Polygon(), 0.0))[0]
                attacks_in_catchment = attack_count_method(acled_df, poly, seg_start, seg_end)
                pct = (attacks_in_catchment / total_attacks * 100) if total_attacks > 0 else 0.0
                
                results.append({
                    "Segment Start": seg_start.strftime("%Y-%m-%d"),
                    "Segment End": seg_end.strftime("%Y-%m-%d"),
                    "Hospital": h_name,
                    "Catchment Area (km²)": round(area_km2, 2),
                    "Attacks in Catchment": attacks_in_catchment,
                    "Total Attacks in Segment": total_attacks,
                    "Percentage of Total Attacks": round(pct, 2),
                })
            
            # Save first Nasser segment for HTML map
            if first_nasser_segment_data is None and "nasser" in hosp_substring.lower():
                first_nasser_segment_data = {
                    "open_hospitals": open_hospitals,
                    "catchments": catchments,
                    "seg_start": seg_start,
                    "seg_end": seg_end,
                }
    
    # 5. Output Excel
    out_excel = OUTPUT_DIR / "catchment_calculation_results.xlsx"
    output_method(results, out_excel)
    print(f"\n[5] Results saved to: {out_excel}")
    
    # 6. Output Test HTML Map (first Nasser segment)
    if first_nasser_segment_data:
        out_html = OUTPUT_DIR / "catchment_map_nasser_first_segment.html"
        output_test_html_map(
            first_nasser_segment_data["open_hospitals"],
            first_nasser_segment_data["catchments"],
            first_nasser_segment_data["seg_start"],
            first_nasser_segment_data["seg_end"],
            out_html,
        )
        print(f"\n[6] HTML map saved to: {out_html}")
    else:
        # Fallback: use first segment of any timeline if Nasser not found
        if results:
            seg = results[0]
            open_h = get_open_hospitals_during_segment(
                hospitals_df, availability_timeline,
                datetime.strptime(seg["Segment Start"], "%Y-%m-%d"),
                datetime.strptime(seg["Segment End"], "%Y-%m-%d"),
            )
            catchments = catchment_area_method(open_h, gaza_union)
            out_html = OUTPUT_DIR / "catchment_map_first_segment.html"
            output_test_html_map(
                open_h,
                catchments,
                datetime.strptime(seg["Segment Start"], "%Y-%m-%d"),
                datetime.strptime(seg["Segment End"], "%Y-%m-%d"),
                out_html,
            )
            print(f"\n[6] HTML map (fallback) saved to: {out_html}")
    
    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
