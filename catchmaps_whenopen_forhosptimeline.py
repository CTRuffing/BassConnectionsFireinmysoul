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
from folium.plugins import HeatMap

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
        header_date = _parse_date(raw.iloc[0, 3])
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
# Identify All Status Change Periods
# ---------------------------------------------------------------------------

def get_all_status_change_periods(hospitals_df: pd.DataFrame,
                                 availability_timeline: Dict[str, List[Tuple[datetime, str]]]) -> List[Tuple[datetime, datetime]]:
    """
    Identify all periods where hospital availability configuration remains constant.
    Returns list of (period_start, period_end) tuples.
    """
    # Collect all unique status change dates across all hospitals
    all_change_dates = set()
    
    for hosp_name in hospitals_df["Hospital"]:
        if hosp_name in availability_timeline:
            for dt, _ in availability_timeline[hosp_name]:
                all_change_dates.add(dt)
    
    # Sort dates and create periods
    sorted_dates = sorted(all_change_dates)
    
    if not sorted_dates:
        return []
    
    periods = []
    for i in range(len(sorted_dates) - 1):
        period_start = sorted_dates[i]
        period_end = sorted_dates[i + 1] - timedelta(days=1)
        periods.append((period_start, period_end))
    
    # Add a final period extending far into the future (or you can set a specific end date)
    # For now, let's extend 1 year past the last change date
    final_end = sorted_dates[-1] + timedelta(days=365)
    periods.append((sorted_dates[-1], final_end))
    
    return periods


def get_open_hospitals_at_date(hospitals_df: pd.DataFrame,
                               availability_timeline: Dict[str, List[Tuple[datetime, str]]],
                               check_date: datetime) -> List[Tuple[str, float, float]]:
    """
    Get list of hospitals that are open on a specific date.
    Returns list of (hospital_name, lat, lon).
    """
    open_hospitals = []
    for _, row in hospitals_df.iterrows():
        name = str(row["Hospital"]).strip()
        status = get_hospital_status_at_date(availability_timeline, name, check_date)
        if status == "Open":
            open_hospitals.append((name, float(row["lat"]), float(row["lon"])))
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
    
    # Keep polygons in projected CRS for distance calculations
    polys_proj = gpd.GeoDataFrame(geometry=poly_list, crs=CRS_WEBMERC)
    assigned: List[Tuple[str, Any]] = []
    hosp_pts_proj = hosp_proj.geometry  # Already in projected CRS
    
    for poly in polys_proj.geometry:
        if poly is None or poly.is_empty:
            continue
        rep = poly.representative_point()
        # Calculate distances in projected CRS (meters)
        dists = hosp_pts_proj.distance(rep)
        nearest_idx = int(dists.idxmin())
        hosp_name = hosp_gdf.iloc[nearest_idx]["Hospital"]
        # Convert polygon back to WGS84 for storage
        poly_wgs = gpd.GeoSeries([poly], crs=CRS_WEBMERC).to_crs(CRS_WGS84).iloc[0]
        assigned.append((hosp_name, poly_wgs))
    
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
    
    # Assign leftover to nearest - using projected CRS for distance calculations
    covered = unary_union([g for g in result.values() if g and not g.is_empty])
    leftover = gaza_union.difference(covered) if covered else gaza_union
    if leftover and not leftover.is_empty:
        pieces = [leftover] if isinstance(leftover, Polygon) else list(leftover.geoms)
        # Convert hospital points and leftover pieces to projected CRS for distance calculation
        hosp_pts_proj = hosp_proj.geometry
        for piece in pieces:
            if piece.is_empty:
                continue
            # Convert piece to projected CRS
            piece_proj = gpd.GeoSeries([piece], crs=CRS_WGS84).to_crs(CRS_WEBMERC).iloc[0]
            rep = piece_proj.representative_point()
            dists = hosp_pts_proj.distance(rep)
            nearest_idx = int(dists.idxmin())
            n = hosp_names[nearest_idx]
            cur = result.get(n, Polygon())
            result[n] = cur.union(piece) if cur and not cur.is_empty else piece
    
    return result


# ---------------------------------------------------------------------------
# Attack Count Methods
# ---------------------------------------------------------------------------

def pts_to_heatlist(acled_df: pd.DataFrame, period_start: datetime, period_end: datetime, round_decimals: int = 5) -> Tuple[List[List[float]], float]:
    """
    Convert ACLED DataFrame to HeatMap-style list [[lat, lon, weight], ...]
    for the specified period. Aggregates by rounded coordinates so co-located events increase weight.
    
    Returns:
        (heat_list, max_weight) where max_weight is used as HeatMap max_val.
    """
    # Filter by date range
    mask = (acled_df["_date"] >= period_start.date()) & (acled_df["_date"] <= period_end.date())
    subset = acled_df[mask].copy()
    
    if subset.empty:
        return [], 1.0
    
    # Round coordinates to avoid tiny floating differences for truly colocated events
    subset["lon_r"] = subset["_lon"].round(round_decimals)
    subset["lat_r"] = subset["_lat"].round(round_decimals)
    
    # Group by rounded coordinates and count
    grouped = subset.groupby(["lat_r", "lon_r"]).size().reset_index(name="count")
    
    # Build heat list in the format expected by folium.plugins.HeatMap: [lat, lon, weight]
    heat = grouped.apply(
        lambda r: [float(r["lat_r"]), float(r["lon_r"]), float(r["count"])], 
        axis=1
    ).tolist()
    
    max_weight = float(grouped["count"].max()) if not grouped.empty else 1.0
    return heat, max_weight


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
                       period_start: datetime, period_end: datetime) -> int:
    """Count attacks within catchment polygon during period."""
    mask = (acled_df["_date"] >= period_start.date()) & (acled_df["_date"] <= period_end.date())
    subset = acled_df[mask]
    count = 0
    for _, row in subset.iterrows():
        pt = Point(row["_lon"], row["_lat"])
        if pt.within(polygon):
            count += 1
    return count


# ---------------------------------------------------------------------------
# Output HTML Map
# ---------------------------------------------------------------------------

def output_html_map(open_hospitals: List[Tuple[str, float, float]],
                   catchments: Dict[str, Tuple[Any, float]],
                   attack_counts: Dict[str, int],
                   acled_df: pd.DataFrame,
                   period_start: datetime, period_end: datetime,
                   output_path: Path) -> None:
    """
    Create HTML map for a specific period:
    - Hospital markers with names
    - Catchment area polygons with area size AND attack count in tooltip
    - Heatmap layer showing attack distribution
    """
    if not open_hospitals:
        return
    
    center_lat = sum(h[1] for h in open_hospitals) / len(open_hospitals)
    center_lon = sum(h[2] for h in open_hospitals) / len(open_hospitals)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="OpenStreetMap")
    
    # Add heatmap layer for attacks
    heat_list, max_weight = pts_to_heatlist(acled_df, period_start, period_end)
    if heat_list:
        HeatMap(
            heat_list,
            max_val=max_weight,
            radius=15,
            blur=20,
            max_zoom=13,
            min_opacity=0.3,
            gradient={0.4: 'blue', 0.6: 'lime', 0.7: 'yellow', 0.8: 'orange', 1.0: 'red'}
        ).add_to(m)
    
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
        attacks = attack_counts.get(hosp_name, 0)
        tooltip_text = f"{hosp_name}<br>Area: {area_km2:.2f} km²<br>Attacks: {attacks}"
        
        if isinstance(poly, MultiPolygon):
            for geom in poly.geoms:
                _add_polygon_to_map(m, geom, color, tooltip_text)
        else:
            _add_polygon_to_map(m, poly, color, tooltip_text)
    
    # Calculate total attacks in period
    total_attacks = sum(attack_counts.values())
    
    title_html = f"""
    <div style="position: fixed; top: 10px; left: 50px; width: 400px; padding: 10px;
                background: white; border: 2px solid grey; z-index: 9999; font-size: 14px;">
    <b>Hospital Catchment Areas</b><br>
    Period: {period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}<br>
    Open Hospitals: {len(open_hospitals)}<br>
    Total Attacks in Period: {total_attacks}
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
    """Run the full catchment area pipeline with HTML map generation."""
    print("=" * 70)
    print("Hospital Catchment Area HTML Map Generation")
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
    
    # 4. Identify all status change periods
    print("\n[4] Identifying status change periods...")
    periods = get_all_status_change_periods(hospitals_df, availability_timeline)
    print(f"    Found {len(periods)} distinct periods")
    
    # 5. Generate HTML map for each period
    print("\n[5] Generating HTML maps for each period...")
    map_count = 0
    
    for period_start, period_end in periods:
        # Get hospitals open during this period
        # Check status at the start of the period
        open_hospitals = get_open_hospitals_at_date(
            hospitals_df, availability_timeline, period_start
        )
        
        if not open_hospitals:
            print(f"    Skipping {period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}: No open hospitals")
            continue
        
        # Calculate catchment areas
        catchments = catchment_area_method(open_hospitals, gaza_union)
        
        # Count attacks in each catchment
        attack_counts = {}
        for hosp_name, (poly, area_km2) in catchments.items():
            if poly and not poly.is_empty:
                attacks = attack_count_method(acled_df, poly, period_start, period_end)
                attack_counts[hosp_name] = attacks
            else:
                attack_counts[hosp_name] = 0
        
        # Generate output filename
        start_str = period_start.strftime('%Y%m%d')
        end_str = period_end.strftime('%Y%m%d')
        output_path = OUTPUT_DIR / f"{start_str}_{end_str}_catchmaps.html"
        
        # Create HTML map
        output_html_map(
            open_hospitals,
            catchments,
            attack_counts,
            acled_df,
            period_start,
            period_end,
            output_path
        )
        
        map_count += 1
        print(f"    [{map_count}] Generated: {output_path.name}")
        print(f"         Open hospitals: {len(open_hospitals)}")
        print(f"         Total attacks in period: {sum(attack_counts.values())}")
    
    print(f"\n[6] Complete! Generated {map_count} HTML maps")
    print("=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()