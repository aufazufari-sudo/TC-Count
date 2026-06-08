#!/usr/bin/env python3
"""Interactive North Atlantic PRMSL plotter with hurricane-eye overlay.

Assumptions
- PRMSL NetCDF variable names:
    PRMSL, lat, lon, time
- PRMSL units are Pa and are converted to hPa for plotting
- IBTrACS CSV contains at least:
    ISO_TIME, USA_STATUS, USA_LAT, USA_LON

Whenever USA_STATUS == 'HU', the hurricane eye coordinates from USA_LAT/USA_LON
are highlighted in red on the map for matching timestamps.

Interactive controls
- Left / Right arrow keys: move backward / forward by one time index
- Previous / Next buttons: move backward / forward by one time index
- Slider: jump directly to a time index
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import cartopy.crs as ccrs
import cartopy.feature as cfeature

VAR_NAME = 'PRMSL'
LAT_NAME = 'lat'
LON_NAME = 'lon'
TIME_NAME = 'time'

LAT_MIN, LAT_MAX = 0, 60
LON_MIN, LON_MAX = 260, 350

DEFAULT_NETCDF = 'Data_M1/PRMSL/PRMSL.1980_6hourly.nc'
DEFAULT_TRACK_CSV = 'IBTrACS/ibtracs.NA.list.v04r01.HU.csv'


def open_and_subset_prmsl(path: str) -> xr.DataArray:
    ds = xr.open_dataset(path)

    # Convert -180..180 longitude to 0..360 if needed.
    if float(ds[LON_NAME].min()) < 0:
        ds = ds.assign_coords({LON_NAME: (ds[LON_NAME] + 360) % 360}).sortby(LON_NAME)

    # Handle either ascending or descending latitude.
    lat_slice = slice(LAT_MIN, LAT_MAX)
    if ds[LAT_NAME][0] > ds[LAT_NAME][-1]:
        lat_slice = slice(LAT_MAX, LAT_MIN)

    da = ds[VAR_NAME].sel({
        LAT_NAME: lat_slice,
        LON_NAME: slice(LON_MIN, LON_MAX),
    })

    # Pressure is in Pa; convert to hPa for plotting.
    da = da / 100.0
    da.attrs['units'] = 'hPa'
    return da


def load_hurricane_eyes(csv_path: str) -> pd.DataFrame:
    """Load IBTrACS and keep hurricane eye locations from USA_* columns only."""
    df = pd.read_csv(csv_path, low_memory=False)

    required = ['ISO_TIME', 'USA_STATUS', 'USA_LAT', 'USA_LON']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f'CSV is missing required columns: {missing}')

    df = df[required].copy()
    df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
    df['USA_LAT'] = pd.to_numeric(df['USA_LAT'], errors='coerce')
    df['USA_LON'] = pd.to_numeric(df['USA_LON'], errors='coerce')

    # Keep only rows that are valid hurricanes in the USA_STATUS field.
    df = df[df['USA_STATUS'].astype(str).str.strip().eq('HU')]
    df = df.dropna(subset=['ISO_TIME', 'USA_LAT', 'USA_LON'])

    # Convert longitude to 0..360 to match the PRMSL map domain.
    df.loc[:, 'plot_lon'] = np.where(df['USA_LON'] < 0, df['USA_LON'] + 360, df['USA_LON'])
    df.loc[:, 'plot_lat'] = df['USA_LAT']

    # Keep only points inside the plotted North Atlantic box.
    df = df[
        df['plot_lat'].between(LAT_MIN, LAT_MAX)
        & df['plot_lon'].between(LON_MIN, LON_MAX)
    ].copy()

    # Use second-resolution timestamps for matching with selected map times.
    df.loc[:, 'time_key'] = df['ISO_TIME'].dt.floor('s')
    return df


def format_time(da: xr.DataArray, index: int) -> str:
    return np.datetime_as_string(da[TIME_NAME].values[index], unit='h')


def get_time_key(da: xr.DataArray, time_index: int) -> pd.Timestamp:
    return pd.Timestamp(da[TIME_NAME].values[time_index]).floor('s')


def hurricane_points_at_time(hur_df: pd.DataFrame | None, da: xr.DataArray, time_index: int) -> pd.DataFrame:
    if hur_df is None or hur_df.empty:
        return pd.DataFrame(columns=['plot_lon', 'plot_lat'])
    time_key = get_time_key(da, time_index)
    return hur_df[hur_df['time_key'] == time_key]


def draw_frame(ax, da: xr.DataArray, time_index: int, cmap: str, vmin: float, vmax: float,
               hur_df: pd.DataFrame | None = None):
    ax.clear()
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())

    mesh = ax.pcolormesh(
        da[LON_NAME],
        da[LAT_NAME],
        da.isel({TIME_NAME: time_index}),
        transform=ccrs.PlateCarree(),
        shading='auto',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        zorder=1,
    )

    # Coastline outline only so the pressure remains visible over land.
    ax.coastlines(resolution='110m', linewidth=1.0, color='black', zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='black', zorder=3)

    gl = ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    eye_points = hurricane_points_at_time(hur_df, da, time_index)
    if not eye_points.empty:
        ax.scatter(
            eye_points['plot_lon'],
            eye_points['plot_lat'],
            transform=ccrs.PlateCarree(),
            s=70,
            c='red',
            edgecolors='black',
            linewidths=0.8,
            zorder=5,
            label='Hurricane eye',
        )
        ax.legend(loc='lower left')

    ax.set_title(f'North Atlantic Pressure — {format_time(da, time_index)}')
    return mesh


def interactive_map(da: xr.DataArray, hur_df: pd.DataFrame | None = None,
                    initial_time_index: int = 0, cmap: str = 'viridis') -> None:
    if TIME_NAME not in da.dims:
        raise ValueError(f"'{TIME_NAME}' dimension not found in the data.")

    ntime = da.sizes[TIME_NAME]
    initial_time_index = max(0, min(initial_time_index, ntime - 1))
    vmin = float(da.min())
    vmax = float(da.max())

    fig = plt.figure(figsize=(11.5, 8.5))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    fig.subplots_adjust(bottom=0.22)

    mesh = draw_frame(ax, da, time_index=initial_time_index, cmap=cmap, vmin=vmin, vmax=vmax, hur_df=hur_df)

    cbar = fig.colorbar(mesh, ax=ax, orientation='vertical', pad=0.03)
    cbar.set_label(f"{da.attrs.get('long_name', VAR_NAME)} ({da.attrs.get('units', 'hPa')})")

    slider_ax = fig.add_axes([0.18, 0.09, 0.56, 0.03])
    prev_ax = fig.add_axes([0.18, 0.045, 0.10, 0.035])
    next_ax = fig.add_axes([0.30, 0.045, 0.10, 0.035])

    time_slider = Slider(
        ax=slider_ax,
        label='Time index',
        valmin=0,
        valmax=ntime - 1,
        valinit=initial_time_index,
        valstep=1,
    )

    prev_button = Button(prev_ax, 'Previous')
    next_button = Button(next_ax, 'Next')

    time_text = fig.text(0.18, 0.135, f'Time: {format_time(da, initial_time_index)}', fontsize=10)
    help_text = fig.text(0.47, 0.045, 'Keyboard: Left / Right arrows', fontsize=10)

    def redraw(idx: int) -> None:
        nonlocal mesh
        idx = max(0, min(idx, ntime - 1))
        mesh.remove()
        mesh = draw_frame(ax, da, time_index=idx, cmap=cmap, vmin=vmin, vmax=vmax, hur_df=hur_df)
        time_text.set_text(f'Time: {format_time(da, idx)}')
        fig.canvas.draw_idle()

    def on_slider_change(val):
        redraw(int(time_slider.val))

    def step(delta: int) -> None:
        current = int(time_slider.val)
        new_index = max(0, min(current + delta, ntime - 1))
        if new_index != current:
            time_slider.set_val(new_index)

    def on_prev(event):
        step(-1)

    def on_next(event):
        step(1)

    def on_key(event):
        if event.key == 'left':
            step(-1)
        elif event.key == 'right':
            step(1)

    time_slider.on_changed(on_slider_change)
    prev_button.on_clicked(on_prev)
    next_button.on_clicked(on_next)
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()


def static_map(da: xr.DataArray, hur_df: pd.DataFrame | None = None,
               time_index: int = 0, output: str | None = None, cmap: str = 'viridis') -> None:
    vmin = float(da.min())
    vmax = float(da.max())

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    mesh = draw_frame(ax, da, time_index=time_index, cmap=cmap, vmin=vmin, vmax=vmax, hur_df=hur_df)

    cbar = fig.colorbar(mesh, ax=ax, orientation='vertical', pad=0.03)
    cbar.set_label(f"{da.attrs.get('long_name', VAR_NAME)} ({da.attrs.get('units', 'hPa')})")

    plt.tight_layout()
    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight')
        print(f'Saved figure to: {output}')
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot North Atlantic PRMSL with hurricane-eye overlay from IBTrACS.')
    parser.add_argument('--file', default=DEFAULT_NETCDF, help='Path to PRMSL NetCDF file')
    parser.add_argument('--track-file', default=DEFAULT_TRACK_CSV, help='Path to IBTrACS CSV file')
    parser.add_argument('--time-index', type=int, default=0, help='Initial time index for the plot')
    parser.add_argument('--output', default=None, help='Optional PNG output filename for static plot')
    parser.add_argument('--cmap', default='viridis', help='Matplotlib colormap')
    parser.add_argument('--interactive', action='store_true', help='Open an interactive plot with a time-index slider')
    args = parser.parse_args()

    prmsl_path = Path(args.file)
    if not prmsl_path.exists():
        raise FileNotFoundError(f'PRMSL file not found: {prmsl_path}')

    da = open_and_subset_prmsl(str(prmsl_path))

    hur_df = None
    track_path = Path(args.track_file)
    if track_path.exists():
        hur_df = load_hurricane_eyes(str(track_path))
        print(f'Loaded {len(hur_df)} hurricane-eye points from: {track_path}')
    else:
        print(f'Warning: track file not found, continuing without hurricane overlay: {track_path}')

    if args.interactive:
        interactive_map(da, hur_df=hur_df, initial_time_index=args.time_index, cmap=args.cmap)
    else:
        static_map(da, hur_df=hur_df, time_index=args.time_index, output=args.output, cmap=args.cmap)


if __name__ == '__main__':
    main()
