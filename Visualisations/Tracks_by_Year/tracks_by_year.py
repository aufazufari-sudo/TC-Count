
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# ============================================================
# Configuration
# ============================================================

UZ_FILE = "Visualisations/Tracks_by_Year/UZ_tracks_1980.csv"
UNET_FILE = "Visualisations/Tracks_by_Year/filtered_detections.csv"
IBTRACS_FILE = "Visualisations/Tracks_by_Year/ibtracs.NA.list.v04r01.csv"

# North Atlantic plotting bounds (degrees)
LAT_MIN, LAT_MAX = 0, 70
LON_MIN, LON_MAX = -100, -10  # conventional west-negative longitude
DEFAULT_YEAR = 1980
DEFAULT_SOURCES = ["UZ", "U-Net", "IBTrACS"]

# Marker styling
UZ_COLOR = "#1f77b4"
UNET_COLOR = "#d62728"
IBTRACS_COLOR = "#2ca02c"
IBTRACS_HU_OPACITY = 1.0
IBTRACS_NON_HU_OPACITY = 0.3


# ============================================================
# Helpers
# ============================================================

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def to_west_east_lon(lon_series: pd.Series) -> pd.Series:
    """Convert 0..360 longitude to -180..180 if needed."""
    lon = pd.to_numeric(lon_series, errors="coerce")
    return lon.where(lon <= 180, lon - 360)


def parse_int64(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


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

    out = pd.DataFrame({
        "source": "UZ",
        "year": parse_int64(df["year"]),
        "month": parse_int64(df["month"]),
        "timestamp": df["timestamp"],
        "lat": pd.to_numeric(df["lat"], errors="coerce"),
        "lon": to_west_east_lon(df["lon"]),
        "track_id": df["track_id"] if "track_id" in df.columns else pd.NA,
        "score": pd.NA,
        "usa_status": pd.NA,
        "sid": pd.NA,
        "name": pd.NA,
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

    out = pd.DataFrame({
        "source": "U-Net",
        "year": parse_int64(df["year"]),
        "month": parse_int64(df["month"]),
        "timestamp": df["timestamp"],
        "lat": pd.to_numeric(df["lat"], errors="coerce"),
        "lon": to_west_east_lon(df["lon"]),
        "track_id": pd.NA,
        "score": pd.to_numeric(df["score"], errors="coerce") if "score" in df.columns else pd.NA,
        "usa_status": pd.NA,
        "sid": pd.NA,
        "name": pd.NA,
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

    df = df[df["month"].between(6, 11, inclusive="both")].copy()

    out = pd.DataFrame({
        "source": "IBTrACS",
        "year": parse_int64(df["year"]),
        "month": parse_int64(df["month"]),
        "timestamp": df["timestamp"],
        "lat": df["lat"],
        "lon": df["lon"],
        "track_id": pd.NA,
        "score": pd.NA,
        "usa_status": df["usa_status"],
        "sid": df["SID"] if "SID" in df.columns else pd.NA,
        "name": df["NAME"] if "NAME" in df.columns else pd.NA,
    })

    return out.dropna(subset=["year", "lat", "lon"]).copy()


def _timestamp_str(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d %H:%M").fillna("NA")


def make_empty_figure(message: str) -> go.Figure:
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
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.08,
            xanchor="left",
            x=0,
        ),
        annotations=[
            dict(
                text=message,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=16),
                bgcolor="rgba(255,255,255,0.8)",
            )
        ],
    )
    return fig


def build_figure(df_year: pd.DataFrame, selected_sources: list[str]) -> go.Figure:
    fig = go.Figure()

    if not selected_sources:
        return make_empty_figure("No sources selected.")

    if "UZ" in selected_sources:
        uz = df_year[df_year["source"] == "UZ"].copy()
        if not uz.empty:
            hover_uz = (
                "Source: UZ<br>"
                + "Year: " + uz["year"].astype(str) + "<br>"
                + "Time: " + _timestamp_str(uz["timestamp"]) + "<br>"
                + "Lat: " + uz["lat"].round(2).astype(str) + "<br>"
                + "Lon: " + uz["lon"].round(2).astype(str)
            )
            hover_uz += "<br>Track ID: " + uz["track_id"].astype(str)

            fig.add_trace(
                go.Scattergeo(
                    lon=uz["lon"],
                    lat=uz["lat"],
                    mode="markers",
                    name="UZ",
                    marker=dict(
                        size=7,
                        color=UZ_COLOR,
                        symbol="circle",
                        opacity=0.8,
                        line=dict(width=0.4, color="white"),
                    ),
                    text=hover_uz,
                    hoverinfo="text",
                )
            )

    if "U-Net" in selected_sources:
        unet = df_year[df_year["source"] == "U-Net"].copy()
        if not unet.empty:
            hover_unet = (
                "Source: U-Net<br>"
                + "Year: " + unet["year"].astype(str) + "<br>"
                + "Time: " + _timestamp_str(unet["timestamp"]) + "<br>"
                + "Lat: " + unet["lat"].round(2).astype(str) + "<br>"
                + "Lon: " + unet["lon"].round(2).astype(str)
                + "<br>Score: " + unet["score"].round(3).astype(str)
            )

            fig.add_trace(
                go.Scattergeo(
                    lon=unet["lon"],
                    lat=unet["lat"],
                    mode="markers",
                    name="U-Net",
                    marker=dict(
                        size=7,
                        color=UNET_COLOR,
                        symbol="x",
                        opacity=0.8,
                        line=dict(width=0.6, color="white"),
                    ),
                    text=hover_unet,
                    hoverinfo="text",
                )
            )

    if "IBTrACS" in selected_sources:
        ib = df_year[df_year["source"] == "IBTrACS"].copy()
        if not ib.empty:
            ib_hu = ib[ib["usa_status"].fillna("").str.upper() == "HU"].copy()
            ib_non_hu = ib[ib["usa_status"].fillna("").str.upper() != "HU"].copy()

            if not ib_non_hu.empty:
                hover_ib_non_hu = (
                    "Source: IBTrACS<br>"
                    + "Year: " + ib_non_hu["year"].astype(str) + "<br>"
                    + "Time: " + _timestamp_str(ib_non_hu["timestamp"]) + "<br>"
                    + "Name: " + ib_non_hu["name"].fillna("NA").astype(str) + "<br>"
                    + "SID: " + ib_non_hu["sid"].fillna("NA").astype(str) + "<br>"
                    + "USA_STATUS: " + ib_non_hu["usa_status"].fillna("NA").astype(str) + "<br>"
                    + "Lat: " + ib_non_hu["lat"].round(2).astype(str) + "<br>"
                    + "Lon: " + ib_non_hu["lon"].round(2).astype(str)
                )
                fig.add_trace(
                    go.Scattergeo(
                        lon=ib_non_hu["lon"],
                        lat=ib_non_hu["lat"],
                        mode="markers",
                        name="IBTrACS (non-HU)",
                        marker=dict(
                            size=6,
                            color=IBTRACS_COLOR,
                            symbol="diamond",
                            opacity=IBTRACS_NON_HU_OPACITY,
                            line=dict(width=0.4, color="white"),
                        ),
                        text=hover_ib_non_hu,
                        hoverinfo="text",
                    )
                )

            if not ib_hu.empty:
                hover_ib_hu = (
                    "Source: IBTrACS<br>"
                    + "Year: " + ib_hu["year"].astype(str) + "<br>"
                    + "Time: " + _timestamp_str(ib_hu["timestamp"]) + "<br>"
                    + "Name: " + ib_hu["name"].fillna("NA").astype(str) + "<br>"
                    + "SID: " + ib_hu["sid"].fillna("NA").astype(str) + "<br>"
                    + "USA_STATUS: " + ib_hu["usa_status"].fillna("NA").astype(str) + "<br>"
                    + "Lat: " + ib_hu["lat"].round(2).astype(str) + "<br>"
                    + "Lon: " + ib_hu["lon"].round(2).astype(str)
                )
                fig.add_trace(
                    go.Scattergeo(
                        lon=ib_hu["lon"],
                        lat=ib_hu["lat"],
                        mode="markers",
                        name="IBTrACS (HU)",
                        marker=dict(
                            size=6,
                            color=IBTRACS_COLOR,
                            symbol="diamond",
                            opacity=IBTRACS_HU_OPACITY,
                            line=dict(width=0.4, color="white"),
                        ),
                        text=hover_ib_hu,
                        hoverinfo="text",
                    )
                )

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

    # No figure title: the page already provides the title, avoiding title/legend overlap.
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=80),
        height=760,
        legend=dict(
            title="Visible sources",
            orientation="h",
            yanchor="top",
            y=-0.08,
            xanchor="left",
            x=0,
        ),
    )

    return fig


# ============================================================
# Load and preprocess data
# ============================================================

uz_df = load_uz(UZ_FILE)
unet_df = load_unet(UNET_FILE)
ibtracs_df = load_ibtracs(IBTRACS_FILE)

all_df = pd.concat([uz_df, unet_df, ibtracs_df], ignore_index=True)
all_df = all_df[
    all_df["lat"].between(LAT_MIN, LAT_MAX)
    & all_df["lon"].between(LON_MIN, LON_MAX)
].copy()

available_years = sorted(all_df["year"].dropna().astype(int).unique().tolist())
initial_year = DEFAULT_YEAR if DEFAULT_YEAR in available_years else (available_years[0] if available_years else None)


# ============================================================
# Dash app
# ============================================================

app = Dash(__name__)
app.title = "North Atlantic TC Tracks: UZ vs U-Net vs IBTrACS"

app.layout = html.Div(
    style={
        "fontFamily": "Arial, sans-serif",
        "maxWidth": "1400px",
        "margin": "0 auto",
        "padding": "20px",
    },
    children=[
        html.H2("North Atlantic Tropical Cyclone Track Points"),
        html.P(
            "Filtered to June–November inclusive. Points are not connected into lines. "
            "IBTrACS points are fully opaque at hurricane status (USA_STATUS = HU) and partially transparent otherwise."
        ),
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "320px 1fr",
                "gap": "20px",
                "alignItems": "end",
                "marginBottom": "10px",
            },
            children=[
                html.Div(
                    children=[
                        html.Label("Select year:", style={"fontWeight": "bold", "display": "block", "marginBottom": "6px"}),
                        dcc.Dropdown(
                            id="year-dropdown",
                            options=[{"label": str(y), "value": int(y)} for y in available_years],
                            value=initial_year,
                            clearable=False,
                        ),
                    ]
                ),
                html.Div(
                    children=[
                        html.Label("Visible sources:", style={"fontWeight": "bold", "display": "block", "marginBottom": "6px"}),
                        dcc.Checklist(
                            id="source-checklist",
                            options=[
                                {"label": " UZ", "value": "UZ"},
                                {"label": " U-Net", "value": "U-Net"},
                                {"label": " IBTrACS", "value": "IBTrACS"},
                            ],
                            value=DEFAULT_SOURCES,
                            inline=True,
                            inputStyle={"marginRight": "6px", "marginLeft": "0px"},
                            labelStyle={"marginRight": "18px", "display": "inline-block"},
                        ),
                    ]
                ),
            ],
        ),
        html.Div(id="summary-text", style={"marginBottom": "10px", "fontSize": "14px"}),
        dcc.Graph(id="track-map", config={"displaylogo": False}),
    ],
)


@app.callback(
    Output("track-map", "figure"),
    Output("summary-text", "children"),
    Input("year-dropdown", "value"),
    Input("source-checklist", "value"),
)
def update_map(selected_year, selected_sources):
    selected_sources = selected_sources or []

    if selected_year is None:
        fig = make_empty_figure("No filtered data available.")
        return fig, "No filtered data available."

    df_year = all_df[all_df["year"] == selected_year].copy()
    fig = build_figure(df_year, selected_sources)

    counts = []
    if "UZ" in selected_sources:
        counts.append(f"UZ points: {int((df_year['source'] == 'UZ').sum())}")
    if "U-Net" in selected_sources:
        counts.append(f"U-Net points: {int((df_year['source'] == 'U-Net').sum())}")
    if "IBTrACS" in selected_sources:
        ib_count = int((df_year["source"] == "IBTrACS").sum())
        ib_hu_count = int(((df_year["source"] == "IBTrACS") & (df_year["usa_status"].fillna("").str.upper() == "HU")).sum())
        counts.append(f"IBTrACS points: {ib_count} (HU: {ib_hu_count})")

    if counts:
        summary = f"Selected year: {selected_year} | " + " | ".join(counts)
    else:
        summary = f"Selected year: {selected_year} | No sources selected."

    return fig, summary


if __name__ == "__main__":
    app.run(debug=True)
