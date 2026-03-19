"""
extract_egh_acled.py
--------------------
Extracts ACLED attack events within the European Hospital (EGH) catchment area
between 12/11/2023 and 04/28/2024 (inclusive).

The catchment area geometry changes mid-window as other hospitals open/close,
shifting the Voronoi tessellation. The window is therefore split into 4
sub-periods, each with its own EGH catchment polygon:

  Sub-period 1: 12/11/2023 – 02/19/2024
    Open: Al Shifa, Nasser, European, Kuwait, Al-Aqsa
    (Al-Quds closed 11/5/2023)

  Sub-period 2: 02/20/2024 – 03/17/2024
    Nasser closes 02/20/2024
    Open: Al Shifa, European, Kuwait, Al-Aqsa

  Sub-period 3: 03/18/2024 – 03/31/2024
    Al Shifa closes 03/18/2024
    Open: European, Kuwait, Al-Aqsa

  Sub-period 4: 04/01/2024 – 04/28/2024
    Al Shifa reopens 04/01/2024
    Open: Al Shifa, European, Kuwait, Al-Aqsa

Terminal output: per-sub-period breakdown of EGH catchment area, attacks in
catchment, and total attacks in Gaza during that sub-period, plus a weighted
average catchment area across the full window.

Excel output: EGH_ACLED_20231211_20240428.xlsx — all ACLED rows falling within
the EGH catchment during any of the four sub-periods (all original columns
preserved, plus two added columns: sub_period and catchment_area_km2).

Place this script in the same folder as:
  - ACLED_May_09_25_Gaza.xlsx
  - pse_admin2.geojson
"""

from datetime import datetime, timedelta
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
OUTPUT_PATH           = BASE_PATH / "EGH_ACLED_20231211_20240428.xlsx"

WINDOW_START          = datetime(2023, 12, 11)
WINDOW_END            = datetime(2024,  4, 28)
TARGET_HOSPITAL       = "European Hospital"

CRS_WGS84             = "EPSG:4326"
CRS_WEBMERC           = "EPSG:3857"
CATCHMENT_DISTANCE_KM = 5.0
VORONOI_BUFFER_METERS = 15_000

# ---------------------------------------------------------------------------
# Sub-period definitions
# Each entry: (label, period_start, period_end, open_hospitals_list)
# open_hospitals_list: [(name, lat, lon), ...]
# ---------------------------------------------------------------------------
SUB_PERIODS: List[Dict] = [
    {
        "label"  : "Sub-period 1",
        "start"  : datetime(2023, 12, 11),
        "end"    : datetime(2024,  2, 19),
        "open"   : [
            ("Al Shifa Medical Hospital",  31.52369093, 34.44274363),
            ("Nasser Hospital",            31.34689036, 34.29193795),
            ("European Hospital",          31.303174,   34.31925517),
            ("Kuwait Hospital",            31.28812189, 34.24915904),
            ("Al-Aqsa Hospital",           31.419969,   34.36      ),
            # Al-Quds closed 11/5/2023 — not included
        ],
        "note"   : "Al Shifa open, Nasser open, Al-Quds closed",
    },
    {
        "label"  : "Sub-period 2",
        "start"  : datetime(2024,  2, 20),
        "end"    : datetime(2024,  3, 17),
        "open"   : [
            ("Al Shifa Medical Hospital",  31.52369093, 34.44274363),
            ("European Hospital",          31.303174,   34.31925517),
            ("Kuwait Hospital",            31.28812189, 34.24915904),
            ("Al-Aqsa Hospital",           31.419969,   34.36      ),
            # Nasser closes 2/20/2024
        ],
        "note"   : "Nasser closes 02/20/2024",
    },
    {
        "label"  : "Sub-period 3",
        "start"  : datetime(2024,  3, 18),
        "end"    : datetime(2024,  3, 31),
        "open"   : [
            ("European Hospital",          31.303174,   34.31925517),
            ("Kuwait Hospital",            31.28812189, 34.24915904),
            ("Al-Aqsa Hospital",           31.419969,   34.36      ),
            # Al Shifa closes 3/18/2024
        ],
        "note"   : "Al Shifa closes 03/18/2024",
    },
    {
        "label"  : "Sub-period 4",
        "start"  : datetime(2024,  4,  1),
        "end"    : datetime(2024,  4, 28),
        "open"   : [
            ("Al Shifa Medical Hospital",  31.52369093, 34.44274363),
            ("European Hospital",          31.303174,   34.31925517),
            ("Kuwait Hospital",            31.28812189, 34.24915904),
            ("Al-Aqsa Hospital",           31.419969,   34.36      ),
            # Al Shifa reopens 4/1/2024; Nasser still closed
        ],
        "note"   : "Al Shifa reopens 04/01/2024; Nasser still closed",
    },
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
        polys_for    = [p for (h, p) in assigned if h == hosp]
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
# Catchment geometry for a given set of open hospitals
# ---------------------------------------------------------------------------

def compute_egh_catchment(
    open_hospitals: List[Tuple[str, float, float]],
    gaza_union: Any,
) -> Polygon:
    """Return the EGH catchment polygon (Voronoi ∩ Gaza ∩ 5 km buffer)."""
    hosp_gdf  = gpd.GeoDataFrame(
        {"Hospital": [h[0] for h in open_hospitals]},
        geometry=[Point(h[2], h[1]) for h in open_hospitals],
        crs=CRS_WGS84,
    )
    hosp_proj = hosp_gdf.to_crs(CRS_WEBMERC)

    gaza_proj              = gpd.GeoSeries([gaza_union], crs=CRS_WGS84).to_crs(CRS_WEBMERC)
    minx, miny, maxx, maxy = gaza_proj.total_bounds
    bbox_proj              = box(
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

    clipped = voronoi_poly.intersection(gaza_union)
    if clipped.is_empty:
        raise ValueError("EGH Voronoi region does not intersect Gaza boundary")

    # 5 km geodesic buffer cap
    hosp_row   = hosp_gdf[hosp_gdf["Hospital"] == TARGET_HOSPITAL]
    pt         = hosp_row.geometry.iloc[0]
    geod       = Geod(ellps="WGS84")
    angles     = np.linspace(0, 360, 128)
    circle_pts = []
    for a in angles:
        lon2, lat2, _ = geod.fwd(pt.x, pt.y, a, CATCHMENT_DISTANCE_KM * 1000)
        circle_pts.append((lon2, lat2))
    circle_poly = Polygon(circle_pts)

    catchment = clipped.intersection(circle_poly)
    if catchment.is_empty:
        raise ValueError("EGH catchment is empty after 5 km cap")
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
# Helpers
# ---------------------------------------------------------------------------

def period_days(start: datetime, end: datetime) -> int:
    """Inclusive day count for a period."""
    return (end - start).days + 1


def count_total_attacks_in_period(
    acled_df: pd.DataFrame,
    start: datetime,
    end: datetime,
) -> int:
    mask = (acled_df["_date"] >= start.date()) & (acled_df["_date"] <= end.date())
    return int(mask.sum())


def filter_attacks_in_catchment(
    acled_df: pd.DataFrame,
    catchment: Polygon,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    mask_date = (acled_df["_date"] >= start.date()) & (acled_df["_date"] <= end.date())
    subset    = acled_df[mask_date].copy()
    in_poly   = subset.apply(
        lambda row: Point(row["_lon"], row["_lat"]).within(catchment), axis=1
    )
    return subset[in_poly].copy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    total_window_days = period_days(WINDOW_START, WINDOW_END)

    print("=" * 70)
    print("European Hospital (EGH) ACLED Extraction")
    print(f"Full window : {WINDOW_START.date()} → {WINDOW_END.date()}  "
          f"({total_window_days} days)")
    print("=" * 70)

    print("\n[1] Loading Gaza boundary...")
    gaza_union = load_gaza_boundary(GAZA_GEOJSON)

    print("[2] Loading ACLED data...")
    acled_raw  = load_acled(ACLED_PATH)
    print(f"    Total events in file : {len(acled_raw):,}")

    geod = Geod(ellps="WGS84")

    # Collect results per sub-period
    all_filtered_frames: List[pd.DataFrame] = []
    weighted_area_sum   = 0.0   # Σ (area_km2 × days)
    total_days_sum      = 0     # Σ days  (sanity check)

    print("\n" + "=" * 70)
    print("SUB-PERIOD BREAKDOWN")
    print("=" * 70)

    for sp in SUB_PERIODS:
        label : str                          = sp["label"]
        start : datetime                     = sp["start"]
        end   : datetime                     = sp["end"]
        open_h: List[Tuple[str, float, float]] = sp["open"]
        note  : str                          = sp["note"]

        days = period_days(start, end)

        print(f"\n{label}: {start.date()} → {end.date()}  ({days} days)")
        print(f"  Note       : {note}")
        print(f"  Open hosps : {[h[0] for h in open_h]}")

        # Compute catchment
        catchment  = compute_egh_catchment(open_h, gaza_union)
        area_m2, _ = geod.geometry_area_perimeter(catchment)
        area_km2   = abs(area_m2) / 1e6

        print(f"  EGH catchment area  : {area_km2:.4f} km²")

        # Attacks
        total_attacks = count_total_attacks_in_period(acled_raw, start, end)
        filtered_df   = filter_attacks_in_catchment(acled_raw, catchment, start, end)
        egh_attacks   = len(filtered_df)

        print(f"  Total attacks (Gaza): {total_attacks:,}")
        print(f"  Attacks in EGH area : {egh_attacks:,}")

        # Tag rows with sub-period metadata before accumulating
        filtered_df = filtered_df.copy()
        filtered_df["sub_period"]        = label
        filtered_df["sub_period_dates"]  = f"{start.date()} to {end.date()}"
        filtered_df["catchment_area_km2"] = round(area_km2, 4)

        all_filtered_frames.append(filtered_df)

        # Accumulate for weighted average
        weighted_area_sum += area_km2 * days
        total_days_sum    += days

    # ---------------------------------------------------------------------------
    # Weighted average catchment area
    # ---------------------------------------------------------------------------
    weighted_avg_area = weighted_area_sum / total_days_sum

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Full window                    : {WINDOW_START.date()} → {WINDOW_END.date()}")
    print(f"  Total window days              : {total_window_days}")
    print(f"  Sub-period days accounted for  : {total_days_sum}")

    # Per-period summary table
    print(f"\n  {'Sub-period':<14} {'Days':>6}  {'Area (km²)':>12}  "
          f"{'Weight':>8}  {'Atk/EGH':>9}  {'Atk/Gaza':>9}")
    print(f"  {'-'*14} {'-'*6}  {'-'*12}  {'-'*8}  {'-'*9}  {'-'*9}")

    for sp, frame in zip(SUB_PERIODS, all_filtered_frames):
        days        = period_days(sp["start"], sp["end"])
        weight      = days / total_days_sum
        area_km2    = frame["catchment_area_km2"].iloc[0] if len(frame) else 0.0
        egh_att     = len(frame)
        total_att   = count_total_attacks_in_period(acled_raw, sp["start"], sp["end"])
        print(f"  {sp['label']:<14} {days:>6}  {area_km2:>12.4f}  "
              f"{weight:>8.4f}  {egh_att:>9,}  {total_att:>9,}")

    print(f"\n  Weighted average EGH catchment area: {weighted_avg_area:.4f} km²")
    print(f"  (weighted by days each catchment area was in effect over the full window)")

    # ---------------------------------------------------------------------------
    # Excel output — combine all sub-period filtered rows
    # ---------------------------------------------------------------------------
    combined = pd.concat(all_filtered_frames, ignore_index=True)

    # Move metadata columns to front, then restore original columns (drop _-prefixed helpers)
    meta_cols = ["sub_period", "sub_period_dates", "catchment_area_km2"]
    orig_cols = [c for c in combined.columns if c not in meta_cols + ["_date", "_lat", "_lon"]]
    combined  = combined[meta_cols + orig_cols]

    total_egh_attacks = len(combined)
    print(f"\n  Total EGH catchment attacks (all sub-periods): {total_egh_attacks:,}")

    print(f"\n[3] Writing output to {OUTPUT_PATH.name}...")
    combined.to_excel(OUTPUT_PATH, index=False)
    print("    Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
