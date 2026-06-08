#!/usr/bin/env python3
"""Interactive North Atlantic PRMSL plotter.

Assumes the dataset uses:
- variable: PRMSL
- latitude: lat
- longitude: lon
- time: time
- pressure units: Pa

The map includes a slider so the user can select the time index interactively.
"""

from pathlib import Path
import argparse

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import cartopy.crs as ccrs
import cartopy.feature as cfeature

VAR_NAME = 'PRMSL'
LAT_NAME = 'lat'
LON_NAME = 'lon'
TIME_NAME = 'time'

LAT_MIN, LAT_MAX = 0, 60
LON_MIN, LON_MAX = 260, 350


def open_and_subset(path: str) -> xr.DataArray:
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


def format_time(da: xr.DataArray, index: int) -> str:
    return np.datetime_as_string(da[TIME_NAME].values[index], unit='h')


def draw_frame(ax, da: xr.DataArray, time_index: int, cmap: str, vmin: float, vmax: float):
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

    ax.set_title(f'North Atlantic Pressure — {format_time(da, time_index)}')
    return mesh


def interactive_map(da: xr.DataArray, initial_time_index: int = 0, cmap: str = 'viridis') -> None:
    if TIME_NAME not in da.dims:
        raise ValueError(f"'{TIME_NAME}' dimension not found in the data.")

    ntime = da.sizes[TIME_NAME]
    initial_time_index = max(0, min(initial_time_index, ntime - 1))
    vmin = float(da.min())
    vmax = float(da.max())

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    fig.subplots_adjust(bottom=0.18)

    mesh = draw_frame(ax, da, time_index=initial_time_index, cmap=cmap, vmin=vmin, vmax=vmax)

    cbar = fig.colorbar(mesh, ax=ax, orientation='vertical', pad=0.03)
    cbar.set_label(f"{da.attrs.get('long_name', VAR_NAME)} ({da.attrs.get('units', 'hPa')})")

    slider_ax = fig.add_axes([0.15, 0.07, 0.7, 0.03])
    time_slider = Slider(
        ax=slider_ax,
        label='Time index',
        valmin=0,
        valmax=ntime - 1,
        valinit=initial_time_index,
        valstep=1,
    )

    time_text = fig.text(0.15, 0.11, f'Time: {format_time(da, initial_time_index)}', fontsize=10)

    def update(val):
        nonlocal mesh
        idx = int(time_slider.val)
        mesh.remove()
        mesh = draw_frame(ax, da, time_index=idx, cmap=cmap, vmin=vmin, vmax=vmax)
        time_text.set_text(f'Time: {format_time(da, idx)}')
        fig.canvas.draw_idle()

    time_slider.on_changed(update)
    plt.show()


def static_map(da: xr.DataArray, time_index: int = 0, output: str | None = None, cmap: str = 'viridis') -> None:
    vmin = float(da.min())
    vmax = float(da.max())

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    mesh = draw_frame(ax, da, time_index=time_index, cmap=cmap, vmin=vmin, vmax=vmax)

    cbar = fig.colorbar(mesh, ax=ax, orientation='vertical', pad=0.03)
    cbar.set_label(f"{da.attrs.get('long_name', VAR_NAME)} ({da.attrs.get('units', 'hPa')})")

    plt.tight_layout()
    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight')
        print(f'Saved figure to: {output}')
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot North Atlantic PRMSL from NetCDF.')
    parser.add_argument('--file', default='Data_M1/PRMSL/PRMSL.1980_6hourly.nc', help='Path to NetCDF file')
    parser.add_argument('--time-index', type=int, default=0, help='Initial time index for the plot')
    parser.add_argument('--output', default=None, help='Optional PNG output filename for static plot')
    parser.add_argument('--cmap', default='viridis', help='Matplotlib colormap')
    parser.add_argument('--interactive', action='store_true', help='Open an interactive plot with a time-index slider')
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        raise FileNotFoundError(f'File not found: {path}')

    da = open_and_subset(str(path))

    if args.interactive:
        interactive_map(da, initial_time_index=args.time_index, cmap=args.cmap)
    else:
        static_map(da, time_index=args.time_index, output=args.output, cmap=args.cmap)


if __name__ == '__main__':
    main()
