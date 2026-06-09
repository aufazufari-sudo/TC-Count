#!/usr/bin/env python3
"""Interactive North Atlantic PRMSL plotter with Gaussian TC-label contour overlay.

Cleaned version of the Gaussian-label viewer:
- PRMSL is shown as a filled field.
- Gaussian labels are shown as contour lines only for readability.
- Redundant logic for filled label contours and label colorbars has been removed.

Assumptions
-----------
- PRMSL NetCDF variable names default to: PRMSL, lat, lon, time
- PRMSL is assumed to be in Pa and converted to hPa for plotting when values look like Pa
- North Atlantic map extent defaults to 0-60N and 260-350E
- The label file contains a variable such as tc_label on the same time/lat/lon grid

Interactive controls
--------------------
- Left / Right arrow keys: move backward / forward by one time index
- Previous / Next buttons: move backward / forward by one time index
- Slider: jump directly to a time index
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.widgets import Button, Slider

VAR_NAME = 'PRMSL'
LABEL_VAR_NAME = 'tc_label'
LAT_NAME = 'lat'
LON_NAME = 'lon'
TIME_NAME = 'time'

LAT_MIN, LAT_MAX = 0, 60
LON_MIN, LON_MAX = 260, 350

DEFAULT_NETCDF = 'Data_M1/PRMSL/PRMSL.1980_6hourly.nc'
DEFAULT_LABEL_NETCDF = 'IBTrACS/LABELS.1980_6hourly.nc'


def maybe_to_hpa(da: xr.DataArray) -> xr.DataArray:
    """Convert pressure from Pa to hPa if values look like Pa."""
    vals = da.values
    finite = np.isfinite(vals)
    if finite.any() and float(np.nanmedian(vals[finite])) > 2000:
        return da / 100.0
    return da


def open_and_subset_field(
    path: str,
    var_name: str,
    lat_name: str,
    lon_name: str,
    time_name: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> xr.DataArray:
    """Open a variable from NetCDF and subset to the North Atlantic domain."""
    ds = xr.open_dataset(path)
    if var_name not in ds:
        raise KeyError(f"Variable {var_name!r} not found in {path}.")

    da = ds[var_name]
    for name in (lat_name, lon_name, time_name):
        if name not in da.coords and name not in da.dims:
            raise KeyError(
                f"Coordinate/dimension {name!r} not found for variable {var_name!r} in {path}."
            )

    if da[lat_name].ndim == 1:
        lat_ascending = float(da[lat_name][0]) <= float(da[lat_name][-1])
        lat_slice = slice(lat_min, lat_max) if lat_ascending else slice(lat_max, lat_min)
        da = da.sel({lat_name: lat_slice})

    if da[lon_name].ndim == 1:
        if lon_min <= lon_max:
            da = da.sel({lon_name: slice(lon_min, lon_max)})
        else:
            left = da.sel({lon_name: slice(lon_min, None)})
            right = da.sel({lon_name: slice(None, lon_max)})
            da = xr.concat([left, right], dim=lon_name)

    return da


def align_label_to_field(
    field_da: xr.DataArray,
    label_da: xr.DataArray,
    time_name: str,
    lat_name: str,
    lon_name: str,
) -> xr.DataArray:
    """Align label coordinates to the already-subset PRMSL field."""
    if field_da[lat_name].ndim == 1 and label_da[lat_name].ndim == 1:
        label_da = label_da.sel({lat_name: field_da[lat_name]})
    if field_da[lon_name].ndim == 1 and label_da[lon_name].ndim == 1:
        label_da = label_da.sel({lon_name: field_da[lon_name]})
    label_da = label_da.sel({time_name: field_da[time_name]})
    return label_da


def format_time(da: xr.DataArray, index: int, time_name: str) -> str:
    return np.datetime_as_string(da[time_name].values[index], unit='h')


def add_map_features(ax) -> None:
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='110m', linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4)
    ax.add_feature(cfeature.LAND, facecolor='0.92', zorder=0)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False


def compute_label_levels(label_t: xr.DataArray, label_threshold: float, contour_levels: int) -> np.ndarray | None:
    """Return contour levels above threshold, or None if nothing should be drawn."""
    data = np.asarray(label_t.values)
    finite = np.isfinite(data)
    if not finite.any():
        return None

    max_val = float(np.nanmax(data))
    if max_val < label_threshold:
        return None

    if contour_levels < 1:
        raise ValueError('contour_levels must be at least 1.')

    if contour_levels == 1:
        return np.array([label_threshold])

    return np.linspace(label_threshold, max_val, contour_levels)


def draw_frame(
    ax,
    field_da: xr.DataArray,
    label_da: xr.DataArray | None,
    time_index: int,
    cmap: str,
    vmin: float,
    vmax: float,
    label_color: str,
    label_threshold: float,
    contour_levels: int,
    contour_linewidth: float,
    time_name: str,
    lat_name: str,
    lon_name: str,
    title_prefix: str = 'North Atlantic PRMSL with Gaussian TC Labels',
):
    """Draw one frame and return the PRMSL mappable."""
    ax.clear()
    add_map_features(ax)

    field_t = field_da.isel({time_name: time_index})
    mesh = ax.pcolormesh(
        field_t[lon_name],
        field_t[lat_name],
        field_t,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading='auto',
        transform=ccrs.PlateCarree(),
        zorder=1,
    )

    if label_da is not None:
        label_t = label_da.isel({time_name: time_index})
        levels = compute_label_levels(label_t, label_threshold, contour_levels)
        if levels is not None:
            ax.contour(
                label_t[lon_name],
                label_t[lat_name],
                label_t,
                levels=levels,
                colors=label_color,
                linewidths=contour_linewidth,
                transform=ccrs.PlateCarree(),
                zorder=3,
            )

    ax.set_title(f'{title_prefix}\nTime: {format_time(field_da, time_index, time_name)}')
    return mesh


def interactive_map(
    field_da: xr.DataArray,
    label_da: xr.DataArray | None = None,
    initial_time_index: int = 0,
    cmap: str = 'viridis',
    label_color: str = 'red',
    label_threshold: float = 0.05,
    contour_levels: int = 6,
    contour_linewidth: float = 1.2,
    time_name: str = TIME_NAME,
    lat_name: str = LAT_NAME,
    lon_name: str = LON_NAME,
) -> None:
    if time_name not in field_da.dims:
        raise ValueError(f"{time_name!r} dimension not found in the PRMSL data.")
    if label_da is not None and time_name not in label_da.dims:
        raise ValueError(f"{time_name!r} dimension not found in the label data.")

    ntime = field_da.sizes[time_name]
    if not (0 <= initial_time_index < ntime):
        raise IndexError(f'Initial time index {initial_time_index} out of bounds for 0..{ntime - 1}.')

    vmin = float(field_da.min())
    vmax = float(field_da.max())

    fig = plt.figure(figsize=(12, 7))
    ax = plt.axes([0.08, 0.18, 0.84, 0.74], projection=ccrs.PlateCarree())
    mesh = draw_frame(
        ax=ax,
        field_da=field_da,
        label_da=label_da,
        time_index=initial_time_index,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        label_color=label_color,
        label_threshold=label_threshold,
        contour_levels=contour_levels,
        contour_linewidth=contour_linewidth,
        time_name=time_name,
        lat_name=lat_name,
        lon_name=lon_name,
    )

    cax = fig.add_axes([0.08, 0.10, 0.84, 0.03])
    cb = fig.colorbar(mesh, cax=cax, orientation='horizontal')
    cb.set_label('PRMSL (hPa)')

    ax_prev = fig.add_axes([0.08, 0.03, 0.10, 0.045])
    ax_next = fig.add_axes([0.19, 0.03, 0.10, 0.045])
    ax_slider = fig.add_axes([0.36, 0.03, 0.56, 0.045])

    btn_prev = Button(ax_prev, 'Previous')
    btn_next = Button(ax_next, 'Next')
    slider = Slider(ax_slider, 'Time', 0, ntime - 1, valinit=initial_time_index, valstep=1)

    state = {'idx': int(initial_time_index)}

    def redraw(i: int) -> None:
        draw_frame(
            ax=ax,
            field_da=field_da,
            label_da=label_da,
            time_index=i,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            label_color=label_color,
            label_threshold=label_threshold,
            contour_levels=contour_levels,
            contour_linewidth=contour_linewidth,
            time_name=time_name,
            lat_name=lat_name,
            lon_name=lon_name,
        )
        fig.canvas.draw_idle()

    def on_slider(_val):
        state['idx'] = int(slider.val)
        redraw(state['idx'])

    def step(delta: int):
        new_idx = max(0, min(ntime - 1, state['idx'] + delta))
        if new_idx != state['idx']:
            state['idx'] = new_idx
            slider.set_val(new_idx)

    def on_prev(_event):
        step(-1)

    def on_next(_event):
        step(1)

    def on_key(event):
        if event.key == 'left':
            step(-1)
        elif event.key == 'right':
            step(1)

    slider.on_changed(on_slider)
    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()


def static_map(
    field_da: xr.DataArray,
    label_da: xr.DataArray | None = None,
    time_index: int = 0,
    output: str | None = None,
    cmap: str = 'viridis',
    label_color: str = 'red',
    label_threshold: float = 0.05,
    contour_levels: int = 6,
    contour_linewidth: float = 1.2,
    time_name: str = TIME_NAME,
    lat_name: str = LAT_NAME,
    lon_name: str = LON_NAME,
) -> None:
    vmin = float(field_da.min())
    vmax = float(field_da.max())

    fig = plt.figure(figsize=(12, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())
    mesh = draw_frame(
        ax=ax,
        field_da=field_da,
        label_da=label_da,
        time_index=time_index,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        label_color=label_color,
        label_threshold=label_threshold,
        contour_levels=contour_levels,
        contour_linewidth=contour_linewidth,
        time_name=time_name,
        lat_name=lat_name,
        lon_name=lon_name,
    )

    cb = fig.colorbar(mesh, ax=ax, orientation='vertical', pad=0.02, shrink=0.85)
    cb.set_label('PRMSL (hPa)')

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=150, bbox_inches='tight')
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Plot North Atlantic PRMSL with Gaussian tropical-cyclone label contour overlays.'
    )
    parser.add_argument('--file', default=DEFAULT_NETCDF, help='Path to the PRMSL NetCDF file')
    parser.add_argument('--label-file', default=DEFAULT_LABEL_NETCDF, help='Path to the Gaussian-label NetCDF file')
    parser.add_argument('--var-name', default=VAR_NAME, help='PRMSL variable name')
    parser.add_argument('--label-var-name', default=LABEL_VAR_NAME, help='Gaussian label variable name')
    parser.add_argument('--lat-name', default=LAT_NAME, help='Latitude coordinate name')
    parser.add_argument('--lon-name', default=LON_NAME, help='Longitude coordinate name')
    parser.add_argument('--time-name', default=TIME_NAME, help='Time coordinate name')
    parser.add_argument('--time-index', type=int, default=0, help='Initial time index for plotting')
    parser.add_argument('--output', default=None, help='Optional PNG output filename for static plot')
    parser.add_argument('--cmap', default='viridis', help='Matplotlib colormap for PRMSL')
    parser.add_argument('--label-color', default='red', help='Contour line color for Gaussian labels')
    parser.add_argument('--label-threshold', type=float, default=0.05, help='Draw contours only for label values >= this threshold')
    parser.add_argument('--contour-levels', type=int, default=6, help='Number of Gaussian contour levels to draw')
    parser.add_argument('--contour-linewidth', type=float, default=1.2, help='Line width for Gaussian contours')
    parser.add_argument('--interactive', action='store_true', help='Open an interactive plot with a time slider')
    args = parser.parse_args()

    field_da = open_and_subset_field(
        path=args.file,
        var_name=args.var_name,
        lat_name=args.lat_name,
        lon_name=args.lon_name,
        time_name=args.time_name,
        lat_min=LAT_MIN,
        lat_max=LAT_MAX,
        lon_min=LON_MIN,
        lon_max=LON_MAX,
    )
    field_da = maybe_to_hpa(field_da)

    label_da = open_and_subset_field(
        path=args.label_file,
        var_name=args.label_var_name,
        lat_name=args.lat_name,
        lon_name=args.lon_name,
        time_name=args.time_name,
        lat_min=LAT_MIN,
        lat_max=LAT_MAX,
        lon_min=LON_MIN,
        lon_max=LON_MAX,
    )
    label_da = align_label_to_field(
        field_da=field_da,
        label_da=label_da,
        time_name=args.time_name,
        lat_name=args.lat_name,
        lon_name=args.lon_name,
    )

    if args.interactive:
        interactive_map(
            field_da=field_da,
            label_da=label_da,
            initial_time_index=args.time_index,
            cmap=args.cmap,
            label_color=args.label_color,
            label_threshold=args.label_threshold,
            contour_levels=args.contour_levels,
            contour_linewidth=args.contour_linewidth,
            time_name=args.time_name,
            lat_name=args.lat_name,
            lon_name=args.lon_name,
        )
    else:
        static_map(
            field_da=field_da,
            label_da=label_da,
            time_index=args.time_index,
            output=args.output,
            cmap=args.cmap,
            label_color=args.label_color,
            label_threshold=args.label_threshold,
            contour_levels=args.contour_levels,
            contour_linewidth=args.contour_linewidth,
            time_name=args.time_name,
            lat_name=args.lat_name,
            lon_name=args.lon_name,
        )


if __name__ == '__main__':
    main()
