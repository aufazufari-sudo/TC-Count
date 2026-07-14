#!/usr/bin/env python3
"""Generate Gaussian-label NetCDF files for tropical cyclone centres from IBTrACS.

This script reads one or more reanalysis NetCDF files (e.g. PRMSL fields on a lat/lon/time
regular grid), matches each analysis time to IBTrACS storm-centre positions, and writes a
label cube with the same (time, lat, lon) coordinates. Instead of a single positive pixel,
for each storm centre it places a 2D Gaussian blob:

    G(lat, lon) = exp(-0.5 * [((lat-lat0)/sigma_lat)^2 + (dlon/sigma_lon)^2])

If more than one storm exists at the same time, blobs are combined with max() by default.

Outputs are written to the directory given by --output-dir (default: IBTrACS/).

Examples
--------
Single file:
    python generate_gaussian_ibtracs_labels.py \
        --nc-files Data_M1/PRMSL/PRMSL.1980_6hourly.nc

Multiple files with a glob:
    python generate_gaussian_ibtracs_labels.py \
        --nc-files 'Data_M1/PRMSL/PRMSL.*_6hourly.nc'

Use a narrower blob and additive merge:
    python generate_gaussian_ibtracs_labels.py \
        --nc-files 'Data_M1/PRMSL/PRMSL.*_6hourly.nc' \
        --sigma-lat-deg 1.5 --sigma-lon-deg 1.5 --merge-mode sum
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_TRACK_CSV = 'IBTrACS/ibtracs.NA.list.v04r01.HU.csv'
TIME_NAME = 'time'
LAT_NAME = 'lat'
LON_NAME = 'lon'


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Generate Gaussian blob label NetCDF files from IBTrACS storm centres.'
    )
    p.add_argument(
        '--nc-files', nargs='+', required=True,
        help='One or more NetCDF file paths and/or glob patterns.'
    )
    p.add_argument(
        '--track-file', default=DEFAULT_TRACK_CSV,
        help='Path to the IBTrACS CSV file.'
    )
    p.add_argument('--var-name', default='gaussian_labels', help='Main data variable name (used only for metadata checks).')
    p.add_argument('--time-name', default=TIME_NAME, help='Time coordinate name in the NetCDF files.')
    p.add_argument('--lat-name', default=LAT_NAME, help='Latitude coordinate name in the NetCDF files.')
    p.add_argument('--lon-name', default=LON_NAME, help='Longitude coordinate name in the NetCDF files.')

    p.add_argument('--time-col', default='ISO_TIME', help='IBTrACS time column.')
    p.add_argument('--status-col', default='USA_STATUS', help='IBTrACS status column.')
    p.add_argument('--lat-col', default='USA_LAT', help='IBTrACS latitude column.')
    p.add_argument('--lon-col', default='USA_LON', help='IBTrACS longitude column.')
    p.add_argument(
        '--status-filter', default='HU',
        help='Keep only rows whose status column equals this value. Use ALL to disable filtering.'
    )

    p.add_argument('--sigma-lat-deg', type=float, default=2.0, help='Gaussian sigma in latitude degrees.')
    p.add_argument('--sigma-lon-deg', type=float, default=2.0, help='Gaussian sigma in longitude degrees.')
    p.add_argument(
        '--merge-mode', choices=['max', 'sum'], default='max',
        help='How to combine multiple storm blobs valid at the same time step.'
    )
    p.add_argument(
        '--time-tolerance', default='0s',
        help='Optional nearest-time tolerance for matching tracks to NetCDF times, e.g. 30min, 1h. Default: exact match.'
    )
    p.add_argument(
        '--amplitude', type=float, default=1.0,
        help='Peak Gaussian amplitude at the storm centre.'
    )
    p.add_argument(
        '--dtype', choices=['float32', 'float64'], default='float32',
        help='Data type of the output label array.'
    )
    p.add_argument(
        '--compress', action='store_true',
        help='Enable NetCDF zlib compression when writing files.'
    )
    p.add_argument(
        '--output-dir', default='IBTrACS',
        help='Directory where label NetCDF files will be written.'
    )
    p.add_argument(
        '--output-suffix', default='.nc',
        help='Suffix appended to each input file stem for the output label NetCDF.'
    )
    return p.parse_args()


def expand_input_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in patterns:
        p = Path(item)
        if any(ch in item for ch in '*?[]'):
            paths.extend(sorted(Path().glob(item)))
        elif p.exists():
            paths.append(p)
    # deduplicate while preserving order
    out = []
    seen = set()
    for p in paths:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    if not out:
        raise FileNotFoundError('No NetCDF files matched the provided --nc-files paths/patterns.')
    return out


def load_ibtracs(csv_path: str, time_col: str, status_col: str, lat_col: str, lon_col: str,
                 status_filter: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    required = [time_col, lat_col, lon_col]
    if status_filter != 'ALL':
        required.append(status_col)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f'Missing required IBTrACS columns: {missing}')

    work = df.copy()
    work[time_col] = pd.to_datetime(work[time_col], utc=False, errors='coerce')
    work[lat_col] = pd.to_numeric(work[lat_col], errors='coerce')
    work[lon_col] = pd.to_numeric(work[lon_col], errors='coerce')
    work = work.dropna(subset=[time_col, lat_col, lon_col])

    if status_filter != 'ALL':
        work = work[work[status_col].astype(str).str.upper() == status_filter.upper()].copy()

    # Normalise timestamps to second precision for exact-match comparisons.
    work['time_key'] = work[time_col].dt.floor('s')
    return work[[time_col, 'time_key', lat_col, lon_col] + ([status_col] if status_col in work.columns else [])]


def normalize_longitudes_to_dataset(lons_deg_e: np.ndarray, ds_lons: np.ndarray) -> np.ndarray:
    """Convert storm longitudes to the same convention as the dataset longitudes."""
    lons = np.asarray(lons_deg_e, dtype=float).copy()
    ds_min = float(np.nanmin(ds_lons))
    ds_max = float(np.nanmax(ds_lons))

    if ds_min >= 0 and ds_max > 180:
        # Dataset uses [0, 360) or similar.
        lons = np.mod(lons, 360.0)
    else:
        # Dataset uses [-180, 180] or similar.
        lons = ((lons + 180.0) % 360.0) - 180.0
    return lons


def wrapped_lon_distance(lon_grid: np.ndarray, lon0: float) -> np.ndarray:
    """Shortest signed angular distance in degrees on a circle."""
    return ((lon_grid - lon0 + 180.0) % 360.0) - 180.0


def make_gaussian_blob(lat_vals: np.ndarray,
                       lon_vals: np.ndarray,
                       lat0: float,
                       lon0: float,
                       sigma_lat_deg: float,
                       sigma_lon_deg: float,
                       amplitude: float = 1.0) -> np.ndarray:
    if sigma_lat_deg <= 0 or sigma_lon_deg <= 0:
        raise ValueError('Both sigma values must be > 0.')

    y = (lat_vals - lat0) / sigma_lat_deg
    x = wrapped_lon_distance(lon_vals, lon0) / sigma_lon_deg

    # Robustly build 2D grids whether coordinates are 1D or already 2D.
    if lat_vals.ndim == 1 and lon_vals.ndim == 1:
        yy = y[:, None]
        xx = x[None, :]
    elif lat_vals.ndim == 2 and lon_vals.ndim == 2:
        yy = y
        xx = x
    else:
        raise ValueError('lat/lon coordinates must both be 1D or both be 2D.')

    return amplitude * np.exp(-0.5 * (yy ** 2 + xx ** 2))


def get_lat_lon_coords(ds: xr.Dataset | xr.DataArray, lat_name: str, lon_name: str) -> tuple[np.ndarray, np.ndarray]:
    if lat_name not in ds.coords or lon_name not in ds.coords:
        raise KeyError(f'Dataset must contain coordinates {lat_name!r} and {lon_name!r}.')
    lat_vals = ds[lat_name].values
    lon_vals = ds[lon_name].values
    return lat_vals, lon_vals


def build_time_index_map(times: np.ndarray) -> dict[pd.Timestamp, int]:
    tkeys = pd.to_datetime(times).floor('s')
    return {pd.Timestamp(t): i for i, t in enumerate(tkeys)}


def nearest_time_indices(ds_times: np.ndarray, track_times: pd.Series, tolerance: str) -> np.ndarray:
    ds_index = pd.DatetimeIndex(pd.to_datetime(ds_times).floor('s'))
    t_index = pd.DatetimeIndex(pd.to_datetime(track_times).floor('s'))
    idx = ds_index.get_indexer(t_index, method='nearest', tolerance=pd.Timedelta(tolerance))
    return idx


def generate_labels_for_file(nc_path: Path, track_df: pd.DataFrame, args: argparse.Namespace) -> Path:
    ds = xr.open_dataset(nc_path)
    if args.time_name not in ds.coords and args.time_name not in ds.dims:
        raise KeyError(f'{nc_path}: missing time coordinate {args.time_name!r}.')
    if args.var_name in ds:
        base = ds[args.var_name]
    else:
        # Fall back to the first data variable to recover coordinates.
        first_var = next(iter(ds.data_vars))
        base = ds[first_var]

    lat_vals, lon_vals = get_lat_lon_coords(base, args.lat_name, args.lon_name)
    ds_times = ds[args.time_name].values
    labels = np.zeros((len(ds_times),) + (lat_vals.shape[0], lon_vals.shape[0]) if lat_vals.ndim == 1 else lat_vals.shape,
                      dtype=np.float32 if args.dtype == 'float32' else np.float64)

    # Track subset / time matching.
    local_tracks = track_df.copy()
    local_tracks['norm_lon'] = normalize_longitudes_to_dataset(local_tracks[args.lon_col].to_numpy(), np.asarray(lon_vals))

    if args.time_tolerance == '0s':
        time_to_idx = build_time_index_map(ds_times)
        local_tracks['t_idx'] = local_tracks['time_key'].map(time_to_idx)
    else:
        local_tracks['t_idx'] = nearest_time_indices(ds_times, local_tracks[args.time_col], args.time_tolerance)
        local_tracks.loc[local_tracks['t_idx'] < 0, 't_idx'] = np.nan

    local_tracks = local_tracks.dropna(subset=['t_idx']).copy()
    if local_tracks.empty:
        print(f'No matching storm-centre timestamps found for {nc_path.name}; writing all-zero labels.')

    for t_idx, grp in local_tracks.groupby(local_tracks['t_idx'].astype(int)):
        for _, row in grp.iterrows():
            blob = make_gaussian_blob(
                lat_vals=np.asarray(lat_vals),
                lon_vals=np.asarray(lon_vals),
                lat0=float(row[args.lat_col]),
                lon0=float(row['norm_lon']),
                sigma_lat_deg=args.sigma_lat_deg,
                sigma_lon_deg=args.sigma_lon_deg,
                amplitude=args.amplitude,
            )
            if args.merge_mode == 'max':
                labels[t_idx] = np.maximum(labels[t_idx], blob)
            else:
                labels[t_idx] += blob

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{nc_path.stem.replace("PRMSL", "LABELS")}{args.output_suffix}'

    label_dims = [args.time_name]
    if lat_vals.ndim == 1 and lon_vals.ndim == 1:
        label_dims += [args.lat_name, args.lon_name]
        coords = {
            args.time_name: ds[args.time_name],
            args.lat_name: ds[args.lat_name],
            args.lon_name: ds[args.lon_name],
        }
    else:
        # Curvilinear grid case: reuse coordinate dimensions directly.
        label_dims += list(ds[args.lat_name].dims)
        coords = {args.time_name: ds[args.time_name], args.lat_name: ds[args.lat_name], args.lon_name: ds[args.lon_name]}

    label_da = xr.DataArray(
        labels,
        dims=tuple(label_dims),
        coords=coords,
        name='tc_label',
        attrs={
            'long_name': 'Gaussian tropical cyclone center label',
            'units': '1',
            'description': '2D Gaussian blobs centred on IBTrACS storm centre coordinates for each analysis time.',
            'sigma_lat_deg': float(args.sigma_lat_deg),
            'sigma_lon_deg': float(args.sigma_lon_deg),
            'amplitude': float(args.amplitude),
            'merge_mode': args.merge_mode,
            'source_track_file': str(args.track_file),
            'source_nc_file': str(nc_path),
        },
    )

    out_ds = xr.Dataset({'tc_label': label_da})

    encoding = {}
    if args.compress:
        encoding = {'tc_label': {'zlib': True, 'complevel': 4, 'dtype': args.dtype}}
    else:
        encoding = {'tc_label': {'dtype': args.dtype}}

    out_ds.to_netcdf(out_path, encoding=encoding)
    ds.close()
    return out_path


def main() -> None:
    args = parse_args()
    nc_files = expand_input_paths(args.nc_files)
    track_df = load_ibtracs(
        csv_path=args.track_file,
        time_col=args.time_col,
        status_col=args.status_col,
        lat_col=args.lat_col,
        lon_col=args.lon_col,
        status_filter=args.status_filter,
    )

    print(f'Loaded {len(track_df):,} IBTrACS rows after filtering.')
    print(f'Writing outputs to: {Path(args.output_dir).resolve()}')

    written = []
    for nc_path in nc_files:
        out_path = generate_labels_for_file(nc_path, track_df, args)
        written.append(out_path)
        print(f'Wrote: {out_path}')

    print(f'Done. Generated {len(written)} label file(s).')


if __name__ == '__main__':
    main()
