import math
from datetime import datetime, timedelta
import pandas as pd


def parse_date(x):
    if pd.isna(x):
        return None
    if isinstance(x, datetime):
        return x.date()
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(str(x), fmt).date()
        except Exception:
            continue
    # fallback to pandas
    try:
        return pd.to_datetime(x).date()
    except Exception:
        raise ValueError(f"Unrecognized date format: {x}")


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def guess_cols(df, candidates):
    for c in candidates:
        for col in df.columns:
            if c.lower() in str(col).lower():
                return col
    return None


def load_acled(path):
    ac = pd.read_excel(path)
    date_col = guess_cols(ac, ["event_date", "date", "iso_date", "eventdate"])
    lat_col = guess_cols(ac, ["latitude", "lat", "y"])
    lon_col = guess_cols(ac, ["longitude", "lon", "long", "x"])
    if date_col is None or lat_col is None or lon_col is None:
        raise RuntimeError("Could not find date/lat/lon columns in ACLED file")
    ac["_date"] = pd.to_datetime(ac[date_col]).dt.date
    ac["_lat"] = pd.to_numeric(ac[lat_col], errors="coerce")
    ac["_lon"] = pd.to_numeric(ac[lon_col], errors="coerce")
    ac = ac.dropna(subset=["_lat", "_lon", "_date"]).copy()
    return ac


def load_hospitals(path):
    h = pd.read_excel(path)
    lat_col = guess_cols(h, ["latitude", "lat", "y"])
    lon_col = guess_cols(h, ["longitude", "lon", "long", "x"])
    name_col = guess_cols(h, ["hospital", "name"])
    if lat_col is None or lon_col is None:
        raise RuntimeError("Could not find lat/lon in hospitals file")
    return h, name_col, lat_col, lon_col


def load_catchments(path):
    c = pd.read_excel(path)
    name_col = guess_cols(c, ["hospital", "name"])
    area_col = guess_cols(c, ["area", "km2", "km^2"])
    start_col = guess_cols(c, ["start", "from", "date_start"])
    end_col = guess_cols(c, ["end", "to", "date_end"])
    if area_col is None or start_col is None or end_col is None:
        raise RuntimeError("Could not find area/start/end columns in catchment file")
    # normalize
    c = c[[name_col, area_col, start_col, end_col]].copy()
    c.columns = ["hospital", "area_km2", "start", "end"]
    c["start"] = c["start"].apply(parse_date)
    c["end"] = c["end"].apply(parse_date)
    return c


def area_to_radius_km(area_km2):
    if area_km2 <= 0:
        return 0.0
    return math.sqrt(area_km2 / math.pi)


def count_attacks_in_interval(ac, hosp_lat, hosp_lon, start_date, end_date, area_km2):
    # inclusive of both dates
    mask = (ac["_date"] >= start_date) & (ac["_date"] <= end_date)
    subset = ac.loc[mask]
    if subset.empty:
        return 0
    r = area_to_radius_km(area_km2)
    if r == 0:
        return 0
    # compute distances
    dists = subset.apply(lambda row: haversine_km(hosp_lat, hosp_lon, row["_lat"], row["_lon"]), axis=1)
    return int((dists <= r).sum())


def segments(start_date, end_date, delta_days=14):
    cur = start_date
    while cur <= end_date:
        seg_end = min(end_date, cur + timedelta(days=delta_days-1))
        yield cur, seg_end
        cur = seg_end + timedelta(days=1)


def process_for_hospital(hosp_name, hosp_row, hosp_lat, hosp_lon, catchments_df, ac_df, timeline_start, timeline_end):
    # get catchment rows for this hospital
    ch = catchments_df[catchments_df["hospital"].astype(str).str.contains(str(hosp_name), case=False, na=False)].copy()
    if ch.empty:
        raise RuntimeError(f"No catchment data found for {hosp_name}")
    rows = []
    for seg_start, seg_end in segments(timeline_start, timeline_end, 14):
        total_attacks_in_seg = int(((ac_df["_date"] >= seg_start) & (ac_df["_date"] <= seg_end)).sum())
        # find overlaps with catchment intervals
        attacks_in_catchment_for_seg = 0
        # for each catchment interval of this hospital
        for _, crow in ch.iterrows():
            cs = crow["start"]
            ce = crow["end"]
            if ce is None:
                ce = seg_end
            # overlap
            ostart = max(seg_start, cs)
            oend = min(seg_end, ce)
            if ostart <= oend:
                attacks = count_attacks_in_interval(ac_df, hosp_lat, hosp_lon, ostart, oend, float(crow["area_km2"]))
                attacks_in_catchment_for_seg += attacks
        pct = (attacks_in_catchment_for_seg / total_attacks_in_seg * 100) if total_attacks_in_seg>0 else 0.0
        rows.append({
            "hospital": hosp_name,
            "hosp_lat": hosp_lat,
            "hosp_lon": hosp_lon,
            "catchment_area_km2": None,  # variable within segment; left blank
            "seg_start": seg_start,
            "seg_end": seg_end,
            "attacks_in_catchment": attacks_in_catchment_for_seg,
            "pct_of_attacks_in_catchment": pct,
            "total_attacks_in_segment": total_attacks_in_seg,
        })
    return pd.DataFrame(rows)


def main():
    base = "c:\\Users\\qingl\\OneDrive\\Desktop\\Monona Zhou\\bassproj\\BassConnectionsFireinmysoul\\"
    acled_path = base + "ACLED_May_09_25_Gaza.xlsx"
    catchment_path = base + "catchment_areas_over_time.xlsx"
    hospitals_path = base + "Hospitals_OpenCloseoverTime.xlsx"

    print("Loading files...")
    ac = load_acled(acled_path)
    hosp_df, name_col, lat_col, lon_col = load_hospitals(hospitals_path)
    catch = load_catchments(catchment_path)

    # choose rows: Alshifa row 2 (iloc[1]), EGH row 4 (iloc[3]), Nasser row 6 (iloc[5])
    choices = []
    for idx, label in [(1, "Al shifa"), (3, "EGH"), (5, "Nasser")]:
        if idx < len(hosp_df):
            row = hosp_df.iloc[idx]
            hosp_name = row[name_col] if name_col is not None else label
            lat = float(row[lat_col])
            lon = float(row[lon_col])
            choices.append((hosp_name, row, lat, lon))
        else:
            raise RuntimeError(f"Hospitals file has fewer rows than expected for index {idx}")

    # use timelines from comments in this repo (hardcoded as per attackcalc_incatchmentareas.py)
    timelines = {
        "Al shifa": (parse_date("10/13/2023"), parse_date("11/03/2023")),
        "EGH": (parse_date("12/11/2023"), parse_date("04/28/2024")),
        "Nasser": (parse_date("11/11/2024"), parse_date("02/02/2025")),
    }

    writer = pd.ExcelWriter(base + "perc_attacks_incatchmentareas_results.xlsx", engine="openpyxl")
    for hosp_name, row, lat, lon in choices:
        # match timeline key by containing name
        matched_key = None
        for k in timelines:
            if k.lower() in str(hosp_name).lower() or str(hosp_name).lower() in k.lower():
                matched_key = k
                break
        if matched_key is None:
            print(f"No timeline found for {hosp_name}, skipping")
            continue
        tstart, tend = timelines[matched_key]
        df_out = process_for_hospital(hosp_name, row, lat, lon, catch, ac, tstart, tend)
        df_out.to_excel(writer, sheet_name=matched_key[:31], index=False)

    writer.save()
    print("Results written to perc_attacks_incatchmentareas_results.xlsx")


if __name__ == "__main__":
    main()
