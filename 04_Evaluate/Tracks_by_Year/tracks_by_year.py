
import math
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# ============================================================
# Configuration
# ============================================================

UZ_FILE = "Visualisations/Tracks_by_Year/UZ_tracks_1980_2015.csv"
UNET_FILE = "Visualisations/Tracks_by_Year/stitched_track_points.csv"
IBTRACS_FILE = "Visualisations/Tracks_by_Year/ibtracs.NA.list.v04r01.csv"

# North Atlantic plotting bounds (degrees)
LAT_MIN, LAT_MAX = 0, 70
LON_MIN, LON_MAX = -100, -10  # conventional west-negative longitude
DEFAULT_YEAR = 1980
DEFAULT_RANGE = [1980, 2015]
DEFAULT_SOURCES = ["UZ", "U-Net", "IBTrACS"]
MATCH_ANGLE_DEGREES = 2.0

# Marker styling
UZ_COLOR = "#1f77b4"
UNET_COLOR = "#d62728"
IBTRACS_COLOR = "#2ca02c"
IBTRACS_HU_OPACITY = 1.0
IBTRACS_NON_HU_OPACITY = 0.3

# U-Net track identifier candidates.
# The loader will use the first match found, so if you have validated a specific
# column you can move it to the front of this list if needed.
TRACK_ID_CANDIDATES = [
    "track_id",
    "stitched_track_id",
    "TRACK_ID",
    "storm_id",
    "STORM_ID",
    "system_id",
    "SYSTEM_ID",
    "track",
    "TRACK",
    "sid",
    "SID",
    "cyclone_id",
    "CYCLONE_ID",
    "uid",
    "UID",
]


# ============================================================
# Helpers
# ============================================================

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def to_west_east_lon(lon_series: pd.Series) -> pd.Series:
    lon = pd.to_numeric(lon_series, errors="coerce")
    return lon.where(lon <= 180, lon - 360)


def parse_int64(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def find_track_id_column(df: pd.DataFrame, preferred=None):
    preferred = preferred or TRACK_ID_CANDIDATES
    for col in preferred:
        if col in df.columns:
            return col
    return None


def is_named_storm(name_series: pd.Series) -> pd.Series:
    names = name_series.fillna("").astype(str).str.strip()
    upper = names.str.upper()
    invalid = upper.isin({"", "UNNAMED", "NOT_NAMED", "NOT NAMED", "UNKNOWN", "NAN"})
    invalid = invalid | upper.str.startswith("UNNAMED")
    return ~invalid


def _timestamp_str(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d %H:%M").fillna("NA")


def clamp_default_range(years):
    if not years:
        return [1980, 2015]
    start = max(min(years), DEFAULT_RANGE[0])
    end = min(max(years), DEFAULT_RANGE[1])
    if start > end:
        start = min(years)
        end = max(years)
    return [int(start), int(end)]


def angular_distance_deg(lat1, lon1, lat2, lon2):
    """Great-circle angular distance in degrees."""
    lat1r = math.radians(lat1)
    lon1r = math.radians(lon1)
    lat2r = math.radians(lat2)
    lon2r = math.radians(lon2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon / 2.0) ** 2
    a = min(1.0, max(0.0, a))
    return math.degrees(2.0 * math.asin(math.sqrt(a)))


def build_time_groups(df: pd.DataFrame):
    grouped = {}
    if df.empty:
        return grouped
    for ts, g in df.groupby("timestamp", dropna=True):
        grouped[ts] = list(zip(g["lat"].astype(float), g["lon"].astype(float)))
    return grouped


def point_has_match(lat, lon, timestamp, time_groups, max_angle_deg=MATCH_ANGLE_DEGREES):
    if pd.isna(timestamp):
        return False
    candidates = time_groups.get(timestamp)
    if not candidates:
        return False
    for ref_lat, ref_lon in candidates:
        if angular_distance_deg(lat, lon, ref_lat, ref_lon) <= max_angle_deg:
            return True
    return False


def filter_to_six_hourly(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """
    Retain only 6-hourly records (00, 06, 12, 18 UTC).
    This is applied to IBTrACS so that it is on the same temporal spacing as UZ and U-Net.
    """
    out = df.copy()
    ts = pd.to_datetime(out[timestamp_col], errors="coerce")
    mask = (
        ts.notna()
        & ts.dt.minute.eq(0)
        & ts.dt.second.eq(0)
        & ts.dt.hour.mod(6).eq(0)
    )
    return out.loc[mask].copy()


# ============================================================
# Loaders
# ============================================================

def load_uz(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = clean_columns(df)

    for col in ["year", "month", "day", "hour", "lat", "lon"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if all(col in df.columns for col in ["year", "month", "day", "hour"]):
        df["timestamp"] = pd.to_datetime(
            dict(year=df["year"], month=df["month"], day=df["day"], hour=df["hour"]),
            errors="coerce",
        )
    else:
        df["timestamp"] = pd.NaT

    df = df[df["month"].between(6, 11, inclusive="both")].copy()

    track_col = find_track_id_column(df, ["track_id", "TRACK_ID", "storm_id", "STORM_ID"])
    if track_col is not None:
        storm_key = df[track_col].astype(str)
        track_values = df[track_col]
    else:
        storm_key = (
            parse_int64(df["year"]).astype(str)
            + "_" + pd.to_datetime(df["timestamp"], errors="coerce").dt.strftime("%Y%m%d%H").fillna("NA")
            + "_" + pd.to_numeric(df["lat"], errors="coerce").round(2).astype(str)
            + "_" + pd.to_numeric(df["lon"], errors="coerce").round(2).astype(str)
        )
        track_values = pd.NA

    out = pd.DataFrame({
        "source": "UZ",
        "year": parse_int64(df["year"]),
        "month": parse_int64(df["month"]),
        "timestamp": df["timestamp"],
        "lat": pd.to_numeric(df["lat"], errors="coerce"),
        "lon": to_west_east_lon(df["lon"]),
        "track_id": track_values,
        "storm_key": "UZ_" + storm_key.astype(str),
        "score": pd.NA,
        "usa_status": pd.NA,
        "sid": pd.NA,
        "name": pd.NA,
        "named_storm": True,
        "ever_hu": pd.NA,
    })
    return out.dropna(subset=["year", "lat", "lon"]).copy()


def load_unet(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = clean_columns(df)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        df["timestamp"] = pd.NaT

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    else:
        df["year"] = df["timestamp"].dt.year

    if "month" in df.columns:
        df["month"] = pd.to_numeric(df["month"], errors="coerce")
    else:
        df["month"] = df["timestamp"].dt.month

    df = df[df["month"].between(6, 11, inclusive="both")].copy()

    track_col = find_track_id_column(df)
    if track_col is not None:
        storm_key = df[track_col].astype(str)
        track_values = df[track_col]
    else:
        storm_key = (
            parse_int64(df["year"]).astype(str)
            + "_" + pd.to_datetime(df["timestamp"], errors="coerce").dt.strftime("%Y%m%d%H").fillna("NA")
            + "_" + pd.to_numeric(df["lat"], errors="coerce").round(2).astype(str)
            + "_" + pd.to_numeric(df["lon"], errors="coerce").round(2).astype(str)
        )
        track_values = pd.NA

    out = pd.DataFrame({
        "source": "U-Net",
        "year": parse_int64(df["year"]),
        "month": parse_int64(df["month"]),
        "timestamp": df["timestamp"],
        "lat": pd.to_numeric(df["lat"], errors="coerce"),
        "lon": to_west_east_lon(df["lon"]),
        "track_id": track_values,
        "storm_key": "UNET_" + storm_key.astype(str),
        "score": pd.to_numeric(df["score"], errors="coerce") if "score" in df.columns else pd.NA,
        "usa_status": pd.NA,
        "sid": pd.NA,
        "name": pd.NA,
        "named_storm": True,
        "ever_hu": pd.NA,
    })
    return out.dropna(subset=["year", "lat", "lon"]).copy()


def load_ibtracs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = clean_columns(df)

    if "SEASON" in df.columns:
        season_numeric = pd.to_numeric(df["SEASON"], errors="coerce")
        df = df[season_numeric.notna()].copy()
        df["SEASON"] = season_numeric.loc[df.index]

    if "ISO_TIME" in df.columns:
        df["timestamp"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    else:
        df["timestamp"] = pd.NaT

    # Filter to 6-hourly records before any downstream calculations so that IBTrACS
    # is on the same temporal spacing as UZ and U-Net.
    df = filter_to_six_hourly(df, "timestamp")

    df["year"] = pd.to_numeric(df["SEASON"], errors="coerce") if "SEASON" in df.columns else df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month

    if "USA_LAT" in df.columns and "USA_LON" in df.columns:
        lat_col, lon_col = "USA_LAT", "USA_LON"
    elif "LAT" in df.columns and "LON" in df.columns:
        lat_col, lon_col = "LAT", "LON"
    else:
        raise ValueError("IBTrACS file must contain either USA_LAT/USA_LON or LAT/LON columns.")

    df["lat"] = pd.to_numeric(df[lat_col], errors="coerce")
    df["lon"] = to_west_east_lon(df[lon_col])

    if "USA_STATUS" in df.columns:
        df["usa_status"] = df["USA_STATUS"].astype(str).str.strip().replace({"nan": pd.NA, "": pd.NA})
    else:
        df["usa_status"] = pd.NA

    if "NAME" in df.columns:
        df["name"] = df["NAME"].astype(str).str.strip().replace({"nan": pd.NA, "": pd.NA})
    else:
        df["name"] = pd.NA

    if "SID" in df.columns:
        df["sid"] = df["SID"].astype(str).str.strip()
    else:
        df["sid"] = parse_int64(df["year"]).astype(str) + "_" + df["name"].fillna("NA").astype(str)

    df["named_storm"] = is_named_storm(df["name"])
    df = df[df["month"].between(6, 11, inclusive="both")].copy()

    hu_by_sid = (
        df.assign(is_hu=df["usa_status"].fillna("").str.upper().eq("HU"))
        .groupby("sid", dropna=False)["is_hu"]
        .any()
    )
    df["ever_hu"] = df["sid"].map(hu_by_sid).fillna(False)

    out = pd.DataFrame({
        "source": "IBTrACS",
        "year": parse_int64(df["year"]),
        "month": parse_int64(df["month"]),
        "timestamp": df["timestamp"],
        "lat": df["lat"],
        "lon": df["lon"],
        "track_id": pd.NA,
        "storm_key": "IB_" + df["sid"].astype(str),
        "score": pd.NA,
        "usa_status": df["usa_status"],
        "sid": df["sid"],
        "name": df["name"],
        "named_storm": df["named_storm"],
        "ever_hu": df["ever_hu"],
    })
    return out.dropna(subset=["year", "lat", "lon", "sid"]).copy()


# ============================================================
# Figures and metrics
# ============================================================

def make_empty_map(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_geos(
        projection_type="equirectangular",
        showland=True,
        landcolor="rgb(240,240,240)",
        showocean=True,
        oceancolor="rgb(230,245,255)",
        showcoastlines=True,
        coastlinecolor="gray",
        showcountries=True,
        countrycolor="lightgray",
        lataxis_range=[LAT_MIN, LAT_MAX],
        lonaxis_range=[LON_MIN, LON_MAX],
        resolution=50,
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=80),
        height=760,
        legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="left", x=0),
        annotations=[dict(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=16), bgcolor="rgba(255,255,255,0.8)")],
    )
    return fig


def build_map_figure(df_year: pd.DataFrame, selected_sources: list[str]) -> go.Figure:
    fig = go.Figure()
    if not selected_sources:
        return make_empty_map("No sources selected.")

    if "UZ" in selected_sources:
        uz = df_year[df_year["source"] == "UZ"].copy()
        if not uz.empty:
            hover = (
                "Source: UZ<br>Year: " + uz["year"].astype(str)
                + "<br>Time: " + _timestamp_str(uz["timestamp"])
                + "<br>Lat: " + uz["lat"].round(2).astype(str)
                + "<br>Lon: " + uz["lon"].round(2).astype(str)
                + "<br>Track ID: " + uz["storm_key"].astype(str)
            )
            fig.add_trace(go.Scattergeo(lon=uz["lon"], lat=uz["lat"], mode="markers", name="UZ", marker=dict(size=7, color=UZ_COLOR, symbol="circle", opacity=0.8, line=dict(width=0.4, color="white")), text=hover, hoverinfo="text"))

    if "U-Net" in selected_sources:
        unet = df_year[df_year["source"] == "U-Net"].copy()
        if not unet.empty:
            hover = (
                "Source: U-Net<br>Year: " + unet["year"].astype(str)
                + "<br>Time: " + _timestamp_str(unet["timestamp"])
                + "<br>Lat: " + unet["lat"].round(2).astype(str)
                + "<br>Lon: " + unet["lon"].round(2).astype(str)
                + "<br>Track ID: " + unet["storm_key"].astype(str)
            )
            if unet["score"].notna().any():
                hover += "<br>Score: " + unet["score"].round(3).astype(str)
            fig.add_trace(go.Scattergeo(lon=unet["lon"], lat=unet["lat"], mode="markers", name="U-Net", marker=dict(size=7, color=UNET_COLOR, symbol="x", opacity=0.8, line=dict(width=0.6, color="white")), text=hover, hoverinfo="text"))

    if "IBTrACS" in selected_sources:
        # Map displays named storms only; FA metric still uses full IBTrACS internally.
        ib = df_year[(df_year["source"] == "IBTrACS") & (df_year["named_storm"].fillna(False))].copy()
        if not ib.empty:
            ib_hu = ib[ib["usa_status"].fillna("").str.upper() == "HU"].copy()
            ib_non_hu = ib[ib["usa_status"].fillna("").str.upper() != "HU"].copy()

            if not ib_non_hu.empty:
                hover = (
                    "Source: IBTrACS<br>Year: " + ib_non_hu["year"].astype(str)
                    + "<br>Time: " + _timestamp_str(ib_non_hu["timestamp"])
                    + "<br>Name: " + ib_non_hu["name"].fillna("NA").astype(str)
                    + "<br>SID: " + ib_non_hu["sid"].fillna("NA").astype(str)
                    + "<br>USA_STATUS: " + ib_non_hu["usa_status"].fillna("NA").astype(str)
                    + "<br>Lat: " + ib_non_hu["lat"].round(2).astype(str)
                    + "<br>Lon: " + ib_non_hu["lon"].round(2).astype(str)
                )
                fig.add_trace(go.Scattergeo(lon=ib_non_hu["lon"], lat=ib_non_hu["lat"], mode="markers", name="IBTrACS (named, non-HU)", marker=dict(size=6, color=IBTRACS_COLOR, symbol="diamond", opacity=IBTRACS_NON_HU_OPACITY, line=dict(width=0.4, color="white")), text=hover, hoverinfo="text"))

            if not ib_hu.empty:
                hover = (
                    "Source: IBTrACS<br>Year: " + ib_hu["year"].astype(str)
                    + "<br>Time: " + _timestamp_str(ib_hu["timestamp"])
                    + "<br>Name: " + ib_hu["name"].fillna("NA").astype(str)
                    + "<br>SID: " + ib_hu["sid"].fillna("NA").astype(str)
                    + "<br>USA_STATUS: " + ib_hu["usa_status"].fillna("NA").astype(str)
                    + "<br>Lat: " + ib_hu["lat"].round(2).astype(str)
                    + "<br>Lon: " + ib_hu["lon"].round(2).astype(str)
                )
                fig.add_trace(go.Scattergeo(lon=ib_hu["lon"], lat=ib_hu["lat"], mode="markers", name="IBTrACS (named, HU)", marker=dict(size=6, color=IBTRACS_COLOR, symbol="diamond", opacity=IBTRACS_HU_OPACITY, line=dict(width=0.4, color="white")), text=hover, hoverinfo="text"))

    fig.update_geos(
        projection_type="equirectangular",
        showland=True,
        landcolor="rgb(240,240,240)",
        showocean=True,
        oceancolor="rgb(230,245,255)",
        showcoastlines=True,
        coastlinecolor="gray",
        showcountries=True,
        countrycolor="lightgray",
        lataxis_range=[LAT_MIN, LAT_MAX],
        lonaxis_range=[LON_MIN, LON_MAX],
        resolution=50,
    )
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=80), height=760, legend=dict(title="Visible traces", orientation="h", yanchor="top", y=-0.08, xanchor="left", x=0))
    return fig


def compute_counts_by_year(uz_df: pd.DataFrame, unet_df: pd.DataFrame, ibtracs_df: pd.DataFrame) -> pd.DataFrame:
    named_ib = ibtracs_df[ibtracs_df["named_storm"].fillna(False)].copy()
    years = sorted(
        set(uz_df["year"].dropna().astype(int))
        .union(set(unet_df["year"].dropna().astype(int)))
        .union(set(named_ib["year"].dropna().astype(int)))
    )
    counts = pd.DataFrame(index=years)
    counts["UZ"] = uz_df.dropna(subset=["storm_key"]).groupby(uz_df["year"].astype(int))["storm_key"].nunique()
    counts["U-Net"] = unet_df.dropna(subset=["storm_key"]).groupby(unet_df["year"].astype(int))["storm_key"].nunique()
    counts["IBTrACS_named"] = named_ib.dropna(subset=["sid"]).groupby(named_ib["year"].astype(int))["sid"].nunique()
    hu_storms = named_ib[named_ib["ever_hu"].fillna(False)]
    counts["IBTrACS_HU"] = hu_storms.dropna(subset=["sid"]).groupby(hu_storms["year"].astype(int))["sid"].nunique()
    return counts.fillna(0).astype(int).reset_index().rename(columns={"index": "year"})


def build_count_figure(counts_df: pd.DataFrame, selected_sources: list[str], selected_range: list[int], selected_year: int | None) -> go.Figure:
    fig = go.Figure()
    yr0, yr1 = selected_range
    plot_df = counts_df[(counts_df["year"] >= yr0) & (counts_df["year"] <= yr1)].copy()

    if not selected_sources:
        fig.update_layout(title="TC count by year", margin=dict(l=40, r=20, t=60, b=80), height=420, annotations=[dict(text="No sources selected.", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=16))])
        return fig

    if "UZ" in selected_sources:
        fig.add_trace(go.Scatter(x=plot_df["year"], y=plot_df["UZ"], mode="lines+markers", name="UZ", line=dict(color=UZ_COLOR, width=2), marker=dict(size=7)))
    if "U-Net" in selected_sources:
        fig.add_trace(go.Scatter(x=plot_df["year"], y=plot_df["U-Net"], mode="lines+markers", name="U-Net", line=dict(color=UNET_COLOR, width=2), marker=dict(size=7)))
    if "IBTrACS" in selected_sources:
        fig.add_trace(go.Scatter(x=plot_df["year"], y=plot_df["IBTrACS_named"], mode="lines+markers", name="IBTrACS named storms", line=dict(color=IBTRACS_COLOR, width=2, dash="dash"), marker=dict(size=7, symbol="diamond")))
        fig.add_trace(go.Scatter(x=plot_df["year"], y=plot_df["IBTrACS_HU"], mode="lines+markers", name="IBTrACS storms reaching HU", line=dict(color=IBTRACS_COLOR, width=2), marker=dict(size=7, symbol="diamond")))

    if selected_year is not None and yr0 <= selected_year <= yr1:
        fig.add_vline(x=selected_year, line_width=1, line_dash="dot", line_color="gray")

    fig.update_layout(
        title=f"TC count by year ({yr0}–{yr1})",
        xaxis_title="Year",
        yaxis_title="Storm count",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=90),
        height=420,
        legend=dict(title="Series", orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0),
    )
    return fig


def compute_range_metrics(det_df: pd.DataFrame, ib_all_df: pd.DataFrame, ib_named_df: pd.DataFrame, year_range: list[int]):
    yr0, yr1 = year_range
    det_range = det_df[(det_df["year"] >= yr0) & (det_df["year"] <= yr1)].copy()
    ib_all_range = ib_all_df[(ib_all_df["year"] >= yr0) & (ib_all_df["year"] <= yr1)].copy()
    ib_named_hu_range = ib_named_df[
        (ib_named_df["year"] >= yr0)
        & (ib_named_df["year"] <= yr1)
        & (ib_named_df["usa_status"].fillna("").str.upper() == "HU")
    ].copy()

    det_groups = build_time_groups(det_range)
    ib_all_groups = build_time_groups(ib_all_range)

    # HR_HU: fraction of named IBTrACS HU 6-hourly locations matched by detections
    # at the same timestamp within 2 degrees.
    hu_total = len(ib_named_hu_range)
    hu_hits = 0
    if hu_total > 0:
        hu_hits = int(
            ib_named_hu_range.apply(
                lambda r: point_has_match(r["lat"], r["lon"], r["timestamp"], det_groups),
                axis=1,
            ).sum()
        )
    hr_hu = (hu_hits / hu_total) if hu_total > 0 else None

    # FA_AS: detected storms with no corresponding IBTrACS record (named OR unnamed)
    # at the same time within 2 degrees.
    false_alarm_storms = 0
    total_detected_storms = int(det_range["storm_key"].dropna().nunique())
    if total_detected_storms > 0:
        for _, storm_df in det_range.groupby("storm_key", dropna=True):
            matched = bool(
                storm_df.apply(
                    lambda r: point_has_match(r["lat"], r["lon"], r["timestamp"], ib_all_groups),
                    axis=1,
                ).any()
            )
            if not matched:
                false_alarm_storms += 1
    fa_as = false_alarm_storms

    return {
        "range": (yr0, yr1),
        "total_detected_storms": total_detected_storms,
        "hu_total": hu_total,
        "hu_hits": hu_hits,
        "hr_hu": hr_hu,
        "fa_as": fa_as,
    }


def format_metrics_panel(metrics_dict):
    children = [
        html.H4("Range metrics", style={"marginTop": "20px", "marginBottom": "8px"}),
        html.P(
            "HR_HU is computed as the fraction of named IBTrACS hurricane-status 6-hourly points detected at the same timestamp within 2°. "
            "FA_AS is computed as the number of detected storms with no matching IBTrACS record (named or unnamed) at the same timestamp within 2° anywhere along the track within the selected range.",
            style={"marginTop": "0px", "fontSize": "14px"},
        ),
    ]

    if not metrics_dict:
        children.append(html.P("No detection sources selected for metric calculation."))
        return html.Div(children)

    cards = []
    for source_name, stats in metrics_dict.items():
        hr_text = "NA"
        if stats["hr_hu"] is not None:
            hr_text = f"{stats['hr_hu']:.3f} ({stats['hu_hits']}/{stats['hu_total']} matched HU points)"
        card = html.Div(
            style={
                "border": "1px solid #d9d9d9",
                "borderRadius": "6px",
                "padding": "12px",
                "marginBottom": "10px",
                "backgroundColor": "#fafafa",
            },
            children=[
                html.Strong(source_name),
                html.Div(f"HR_HU: {hr_text}"),
                html.Div(f"FA_AS: {stats['fa_as']} false-alarm storms"),
                html.Div(f"Detected storms in range: {stats['total_detected_storms']}"),
            ],
        )
        cards.append(card)

    children.extend(cards)
    return html.Div(children)


# ============================================================
# Load and preprocess data
# ============================================================

uz_df = load_uz(UZ_FILE)
unet_df = load_unet(UNET_FILE)
ibtracs_full_df = load_ibtracs(IBTRACS_FILE)
ibtracs_named_df = ibtracs_full_df[ibtracs_full_df["named_storm"].fillna(False)].copy()

# Map data uses named IBTrACS only for display.
all_df = pd.concat([uz_df, unet_df, ibtracs_named_df], ignore_index=True)
all_df = all_df[
    all_df["lat"].between(LAT_MIN, LAT_MAX) & all_df["lon"].between(LON_MIN, LON_MAX)
].copy()

counts_by_year_df = compute_counts_by_year(uz_df, unet_df, ibtracs_full_df)

available_years = sorted(all_df["year"].dropna().astype(int).unique().tolist())
initial_year = DEFAULT_YEAR if DEFAULT_YEAR in available_years else (available_years[0] if available_years else None)
initial_range = clamp_default_range(available_years)
range_marks = (
    {int(y): str(int(y)) for y in available_years if ((y - min(available_years)) % 5 == 0 or y in {min(available_years), max(available_years)})}
    if available_years else {1980: "1980", 2015: "2015"}
)


# ============================================================
# Dash app
# ============================================================

app = Dash(__name__)
app.title = "North Atlantic TC Tracks: UZ vs U-Net vs IBTrACS"

app.layout = html.Div(
    style={"fontFamily": "Arial, sans-serif", "maxWidth": "1400px", "margin": "0 auto", "padding": "20px"},
    children=[
        html.H2("North Atlantic Tropical Cyclone Track Points"),
        html.P(
            "Filtered to June–November inclusive. Points are not connected into lines. "
            "IBTrACS is filtered to 6-hourly records (00, 06, 12, 18 UTC) so it is temporally aligned with UZ and U-Net. "
            "The map displays named IBTrACS storms only. For metrics, HR_HU uses named IBTrACS hurricane points, while FA_AS checks against all IBTrACS records (including unnamed storms)."
        ),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "320px 1fr", "gap": "20px", "alignItems": "end", "marginBottom": "10px"},
            children=[
                html.Div(children=[
                    html.Label("Select year:", style={"fontWeight": "bold", "display": "block", "marginBottom": "6px"}),
                    dcc.Dropdown(id="year-dropdown", options=[{"label": str(y), "value": int(y)} for y in available_years], value=initial_year, clearable=False),
                ]),
                html.Div(children=[
                    html.Label("Visible sources:", style={"fontWeight": "bold", "display": "block", "marginBottom": "6px"}),
                    dcc.Checklist(
                        id="source-checklist",
                        options=[{"label": " UZ", "value": "UZ"}, {"label": " U-Net", "value": "U-Net"}, {"label": " IBTrACS", "value": "IBTrACS"}],
                        value=DEFAULT_SOURCES,
                        inline=True,
                        inputStyle={"marginRight": "6px", "marginLeft": "0px"},
                        labelStyle={"marginRight": "18px", "display": "inline-block"},
                    ),
                ]),
            ],
        ),
        html.Div(id="summary-text", style={"marginBottom": "10px", "fontSize": "14px"}),
        dcc.Graph(id="track-map", config={"displaylogo": False}),
        html.H3("TC count by year comparison", style={"marginTop": "24px", "marginBottom": "8px"}),
        html.Label("Select year range for count chart and metrics:", style={"fontWeight": "bold", "display": "block", "marginBottom": "6px"}),
        dcc.RangeSlider(
            id="year-range-slider",
            min=min(available_years) if available_years else 1980,
            max=max(available_years) if available_years else 2015,
            step=1,
            allowCross=False,
            value=initial_range,
            marks=range_marks,
            tooltip={"placement": "bottom", "always_visible": False},
        ),
        html.Div(id="range-label", style={"marginTop": "8px", "marginBottom": "8px", "fontSize": "14px"}),
        dcc.Graph(id="count-chart", config={"displaylogo": False}),
        html.Div(id="range-metrics"),
    ],
)


@app.callback(
    Output("track-map", "figure"),
    Output("count-chart", "figure"),
    Output("summary-text", "children"),
    Output("range-label", "children"),
    Output("range-metrics", "children"),
    Input("year-dropdown", "value"),
    Input("source-checklist", "value"),
    Input("year-range-slider", "value"),
)
def update_dashboard(selected_year, selected_sources, selected_range):
    selected_sources = selected_sources or []
    if not selected_range or len(selected_range) != 2:
        selected_range = initial_range
    selected_range = [int(selected_range[0]), int(selected_range[1])]
    if selected_range[0] > selected_range[1]:
        selected_range = [selected_range[1], selected_range[0]]

    range_label = f"Selected range: {selected_range[0]}–{selected_range[1]}"

    if selected_year is None:
        empty_map = make_empty_map("No filtered data available.")
        empty_count = build_count_figure(counts_by_year_df, selected_sources, selected_range, None)
        metrics_panel = format_metrics_panel({})
        return empty_map, empty_count, "No filtered data available.", range_label, metrics_panel

    df_year = all_df[all_df["year"] == selected_year].copy()
    map_fig = build_map_figure(df_year, selected_sources)
    count_fig = build_count_figure(counts_by_year_df, selected_sources, selected_range, selected_year)

    summary_parts = []
    row = counts_by_year_df[counts_by_year_df["year"] == selected_year]
    if "UZ" in selected_sources:
        summary_parts.append(f"UZ points: {int((df_year['source'] == 'UZ').sum())}")
        if not row.empty:
            summary_parts.append(f"UZ TCs: {int(row['UZ'].iloc[0])}")
    if "U-Net" in selected_sources:
        summary_parts.append(f"U-Net points: {int((df_year['source'] == 'U-Net').sum())}")
        if not row.empty:
            summary_parts.append(f"U-Net TCs: {int(row['U-Net'].iloc[0])}")
    if "IBTrACS" in selected_sources:
        ib_points = int((df_year["source"] == "IBTrACS").sum())
        ib_hu_points = int(((df_year["source"] == "IBTrACS") & (df_year["usa_status"].fillna("").str.upper() == "HU")).sum())
        summary_parts.append(f"IBTrACS named points: {ib_points} (HU points: {ib_hu_points})")
        if not row.empty:
            summary_parts.append(f"IBTrACS named storms: {int(row['IBTrACS_named'].iloc[0])}")
            summary_parts.append(f"IBTrACS storms reaching HU: {int(row['IBTrACS_HU'].iloc[0])}")

    summary_text = f"Selected year: {selected_year} | " + (" | ".join(summary_parts) if summary_parts else "No sources selected.")

    metrics_dict = {}
    if "UZ" in selected_sources:
        metrics_dict["UZ"] = compute_range_metrics(uz_df, ibtracs_full_df, ibtracs_named_df, selected_range)
    if "U-Net" in selected_sources:
        metrics_dict["U-Net"] = compute_range_metrics(unet_df, ibtracs_full_df, ibtracs_named_df, selected_range)
    metrics_panel = format_metrics_panel(metrics_dict)

    return map_fig, count_fig, summary_text, range_label, metrics_panel


if __name__ == "__main__":
    app.run(debug=True)
