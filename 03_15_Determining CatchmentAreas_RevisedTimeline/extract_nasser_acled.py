"""
extract_nasser_acled.py
-----------------------
Extracts ACLED attack events that fall within the Nasser Hospital
catchment area between 11/11/2024 and 02/01/2025 (inclusive).

Catchment geometry is computed identically to catchment_maps.py:
  - Voronoi tessellation of all hospitals open on 11/11/2024
  - Clipped to the Gaza outer boundary
  - Capped to a 5 km geodesic buffer around Nasser Hospital

Hospitals open on 11/11/2024 (derived from HOSPITAL_DEFINITIONS):
  - Al Shifa Medical Hospital  (reopened 9/1/2024)
  - Nasser Hospital            (reopened 5/3/2024)
  - European Hospital          (reopened 8/25/2024, closes 5/13/2025)
  - Kuwait Hospital            (always open)
  NOT open:
  - Al-Quds Hospital           (closed 11/5/2023, reopens 3/19/2025)
  - Al-Aqsa Hospital           (closed 8/1/2024)

Output: Nasser_ACLED_20241111_20250201.xlsx  (same folder as this script)

Place this script in the same folder as:
  - ACLED_May_09_25_Gaza.xlsx
  - pse_admin2.geojson
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon, MultiPoint, box
from shapely.ops import unary_union
from pyproj import Geod

try:
    from shapely.ops import voronoi_diagram
    from shapely import geometry as shapely_geom
    _HAS_SHAPELY_VORONOI = True
except ImportError:
    voronoi_diagram = None
    shapely_geom = None
    _HAS_SHAPELY_VORONOI = False

from scipy.spatial import Voronoi

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_PATH             = Path(__file__).parent.resolve()
ACLED_PATH            = BASE_PATH / "ACLED_May_09_25_Gaza.xlsx"
GAZA_GEOJSON          = BASE_PATH / "pse_admin2.geojson"
OUTPUT_PATH           = BASE_PATH / "Nasser_ACLED_20241111_20250201.xlsx"

PERIOD_START          = datetime(2024, 11, 11)
PERIOD_END            = datetime(2025,  2,  1)
TARGET_HOSPITAL       = "Nasser Hospital"

CRS_WGS84             = "EPSG:4326"
CRS_WEBMERC           = "EPSG:3857"
CATCHMENT_DISTANCE_KM = 5.0
VORONOI_BUFFER_METERS = 15_000

# ---------------------------------------------------------------------------
# Hospitals open on 11/11/2024
# ---------------------------------------------------------------------------
OPEN_HOSPITALS_ON_DATE: List[Tuple[str, float, float]] = [
    # (name, lat, lon)
    ("Al Shifa Medical Hospital",  31.52369093, 34.44274363),
    ("Nasser Hospital",            31.34689036, 34.29193795),
    ("European Hospital",          31.303174,   34.31925517),
    ("Kuwait Hospital",            31.28812189, 34.24915904),
]

# ---------------------------------------------------------------------------
# Gaza boundary
# ---------------------------------------------------------------------------

def load_gaza_boundary(path: Path) -> Any:
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(CRS_WGS84)
    return gdf.to_crs(CRS_WGS84).unary_union


# ---------------------------------------------------------------------------
# Voronoi helpers (identical to catchment_maps.py)
# ---------------------------------------------------------------------------

def _extract_voronoi_polygons(vor, bbox_proj, hosp_proj, hosp_gdf) -> Dict[str, Any]:
    poly_list = []
    if hasattr(vor, "geoms"):
        for g in vor.geoms:
            if isinstance(g, (Polygon, MultiPolygon)):
                poly_list.append(g)
    elif isinstance(vor, (Polygon, MultiPolygon)):
        poly_list = [vor]
    if not poly_list:
        return {}

    polys_proj    = gpd.GeoDataFrame(geometry=poly_list, crs=CRS_WEBMERC)
    hosp_pts_proj = hosp_proj.geometry
    assigned: List[Tuple[str, Any]] = []

    for poly in polys_proj.geometry:
        if poly is None or poly.is_empty:
            continue
        rep         = poly.representative_point()
        dists       = hosp_pts_proj.distance(rep)
        nearest_idx = int(dists.idxmin())
        hosp_name   = hosp_gdf.iloc[nearest_idx]["Hospital"]
        poly_wgs    = gpd.GeoSeries([poly], crs=CRS_WEBMERC).to_crs(CRS_WGS84).iloc[0]
        assigned.append((hosp_name, poly_wgs))

    result: Dict[str, Any] = {}
    for hosp in hosp_gdf["Hospital"].values:
        polys_for   = [p for (h, p) in assigned if h == hosp]
        result[hosp] = unary_union(polys_for) if polys_for else Polygon()
    return result


def _scipy_voronoi_clipped(hosp_proj, hosp_gdf, bbox_proj, gaza_union) -> Dict[str, Any]:
    coords     = np.array([(pt.x, pt.y) for pt in hosp_proj.geometry])
    vor        = Voronoi(coords)
    hosp_names = hosp_gdf["Hospital"].values

    def make_bounded_poly(site: np.ndarray) -> Polygon:
        bbox_coords = np.array(bbox_proj.exterior.coords)
        pts         = np.vstack([site, bbox_coords])
        multipoint  = MultiPoint([tuple(p) for p in pts])
        return multipoint.convex_hull.intersection(bbox_proj)

    proj_polys = []
    for pt_idx, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if not region or -1 in region:
            poly = make_bounded_poly(coords[pt_idx])
        else:
            try:
                verts = [vor.vertices[i] for i in region]
                poly  = Polygon(verts).intersection(bbox_proj)
            except Exception:
                poly  = make_bounded_poly(coords[pt_idx])
        proj_polys.append(poly)

    polys_wgs = gpd.GeoDataFrame(geometry=proj_polys, crs=CRS_WEBMERC).to_crs(CRS_WGS84)
    result: Dict[str, Any] = {}
    for i, hosp_name in enumerate(hosp_names):
        try:
            clipped = polys_wgs.iloc[i].geometry.intersection(gaza_union)
        except Exception:
            clipped = Polygon()
        result[hosp_name] = clipped if clipped and not clipped.is_empty else Polygon()

    covered  = unary_union([g for g in result.values() if g and not g.is_empty])
    leftover = gaza_union.difference(covered) if covered else gaza_union
    if leftover and not leftover.is_empty:
        pieces        = [leftover] if isinstance(leftover, Polygon) else list(leftover.geoms)
        hosp_pts_proj = hosp_proj.geometry
        for piece in pieces:
            if piece.is_empty:
                continue
            piece_proj  = gpd.GeoSeries([piece], crs=CRS_WGS84).to_crs(CRS_WEBMERC).iloc[0]
            rep         = piece_proj.representative_point()
            dists       = hosp_pts_proj.distance(rep)
            nearest_idx = int(dists.idxmin())
            n           = hosp_names[nearest_idx]
            cur         = result.get(n, Polygon())
            result[n]   = cur.union(piece) if cur and not cur.is_empty else piece
    return result


# ---------------------------------------------------------------------------
# Catchment geometry
# ---------------------------------------------------------------------------

def compute_nasser_catchment(
    open_hospitals: List[Tuple[str, float, float]],
    gaza_union: Any,
) -> Polygon:
    """Return the Nasser Hospital catchment polygon (Voronoi ∩ Gaza ∩ 5 km buffer)."""
    hosp_gdf  = gpd.GeoDataFrame(
        {"Hospital": [h[0] for h in open_hospitals]},
        geometry=[Point(h[2], h[1]) for h in open_hospitals],
        crs=CRS_WGS84,
    )
    hosp_proj = hosp_gdf.to_crs(CRS_WEBMERC)

    gaza_proj               = gpd.GeoSeries([gaza_union], crs=CRS_WGS84).to_crs(CRS_WEBMERC)
    minx, miny, maxx, maxy  = gaza_proj.total_bounds
    bbox_proj               = box(
        minx - VORONOI_BUFFER_METERS, miny - VORONOI_BUFFER_METERS,
        maxx + VORONOI_BUFFER_METERS, maxy + VORONOI_BUFFER_METERS,
    )

    if _HAS_SHAPELY_VORONOI:
        try:
            multip    = shapely_geom.MultiPoint([(pt.x, pt.y) for pt in hosp_proj.geometry])
            vor       = voronoi_diagram(multip, envelope=bbox_proj, tolerance=0.0)
            polys_map = _extract_voronoi_polygons(vor, bbox_proj, hosp_proj, hosp_gdf)
        except Exception:
            polys_map = _scipy_voronoi_clipped(hosp_proj, hosp_gdf, bbox_proj, gaza_union)
    else:
        polys_map = _scipy_voronoi_clipped(hosp_proj, hosp_gdf, bbox_proj, gaza_union)

    voronoi_poly = polys_map.get(TARGET_HOSPITAL, Polygon())
    if voronoi_poly is None or voronoi_poly.is_empty:
        raise ValueError(f"Could not compute Voronoi region for {TARGET_HOSPITAL}")

    # Clip to Gaza
    clipped = voronoi_poly.intersection(gaza_union)
    if clipped.is_empty:
        raise ValueError("Nasser Hospital Voronoi region does not intersect Gaza boundary")

    # 5 km geodesic buffer cap
    hosp_row    = hosp_gdf[hosp_gdf["Hospital"] == TARGET_HOSPITAL]
    pt          = hosp_row.geometry.iloc[0]
    geod        = Geod(ellps="WGS84")
    angles      = np.linspace(0, 360, 128)
    circle_pts  = []
    for a in angles:
        lon2, lat2, _ = geod.fwd(pt.x, pt.y, a, CATCHMENT_DISTANCE_KM * 1000)
        circle_pts.append((lon2, lat2))
    circle_poly = Polygon(circle_pts)

    catchment = clipped.intersection(circle_poly)
    if catchment.is_empty:
        raise ValueError("Nasser Hospital catchment is empty after 5 km cap")
    if isinstance(catchment, MultiPolygon):
        catchment = max(catchment.geoms, key=lambda p: p.area)

    return catchment


# ---------------------------------------------------------------------------
# ACLED loader
# ---------------------------------------------------------------------------

def load_acled(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    date_col = lat_col = lon_col = None
    for c in df.columns:
        cl = str(c).lower()
        if not date_col and ("event_date" in cl or "date" in cl or "eventdate" in cl):
            date_col = c
        if not lat_col and ("lat" in cl or "latitude" in cl or cl == "y"):
            lat_col = c
        if not lon_col and ("lon" in cl or "longitude" in cl or cl == "x"):
            lon_col = c
    if not (date_col and lat_col and lon_col):
        raise ValueError(f"Cannot find date/lat/lon columns. Found: {list(df.columns)}")
    df = df.rename(columns={date_col: "_date", lat_col: "_lat", lon_col: "_lon"})
    df["_date"] = pd.to_datetime(df["_date"]).dt.date
    df["_lat"]  = pd.to_numeric(df["_lat"], errors="coerce")
    df["_lon"]  = pd.to_numeric(df["_lon"], errors="coerce")
    return df.dropna(subset=["_date", "_lat", "_lon"]).copy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("Nasser Hospital ACLED Extraction")
    print(f"Period : {PERIOD_START.date()} → {PERIOD_END.date()}")
    print(f"Open hospitals in Voronoi: {[h[0] for h in OPEN_HOSPITALS_ON_DATE]}")
    print("=" * 65)

    print("\n[1] Loading Gaza boundary...")
    gaza_union = load_gaza_boundary(GAZA_GEOJSON)

    print("[2] Computing Nasser Hospital catchment geometry...")
    catchment  = compute_nasser_catchment(OPEN_HOSPITALS_ON_DATE, gaza_union)
    geod       = Geod(ellps="WGS84")
    area_m2, _ = geod.geometry_area_perimeter(catchment)
    print(f"    Catchment area : {abs(area_m2) / 1e6:.3f} km²")

    print("[3] Loading ACLED data...")
    acled_raw  = load_acled(ACLED_PATH)
    print(f"    Total events in file : {len(acled_raw):,}")

    print("[4] Filtering by date range...")
    mask_date    = (
        (acled_raw["_date"] >= PERIOD_START.date()) &
        (acled_raw["_date"] <= PERIOD_END.date())
    )
    acled_period = acled_raw[mask_date].copy()
    print(f"    Events in period     : {len(acled_period):,}")

    print("[5] Filtering by Nasser Hospital catchment polygon...")
    in_catchment   = acled_period.apply(
        lambda row: Point(row["_lon"], row["_lat"]).within(catchment), axis=1
    )
    acled_filtered = acled_period[in_catchment].copy()
    print(f"    Events in catchment  : {len(acled_filtered):,}")

    # Restore original column names
    acled_filtered = acled_filtered.drop(columns=["_date", "_lat", "_lon"])

    print(f"\n[6] Writing output to {OUTPUT_PATH.name}...")
    acled_filtered.to_excel(OUTPUT_PATH, index=False)
    print("    Done.")
    print("=" * 65)


if __name__ == "__main__":
    main()
