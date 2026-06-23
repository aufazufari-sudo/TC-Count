
#!/usr/bin/env python3
"""
Train a U-Net hurricane-core detector from 20CR PRMSL and Gaussian IBTrACS labels.

Input pressure files:
    Data_M1/PRMSL/PRMSL.YYYY_6hourly.nc

Input label files:
    IBTrACS/LABELS.YYYY_6hourly.nc

Assumptions:
    - PRMSL files and LABELS files share time/lat/lon coordinates.
    - Label variable is 'tc_label', as produced by the uploaded Gaussian-label script.
    - Pressure variable is auto-detected unless --pressure-var is provided.
    - Domain is already subset to the North Atlantic.

Default behaviour:
    - Hurricane season only: June-November inclusive.
    - Train years: 1980-1983.
    - Validation year: first available year after the training range.
    - Test year: first remaining available year after training and validation.
    - Loss: weighted BCE with sigmoid output.
    - Input: pressure only.
    - If --cv-folds is provided (>= 2), year-based cross-validation is run and
      out-of-sample predictions are exported across the selected year range.
    - If --resume-cv is also provided, completed folds are skipped and their
      outputs are reused while aggregate CV files are rebuilt.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------

def parse_years(values: Optional[List[int]], default: List[int]) -> List[int]:
    if values is None or len(values) == 0:
        return default
    # Deduplicate while preserving sorted order.
    return sorted(set(int(v) for v in values))


def year_folds(years: Sequence[int], n_folds: int) -> List[List[int]]:
    """
    Split sorted years into approximately equal contiguous folds.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2 for cross-validation.")
    years_sorted = sorted(set(int(y) for y in years))
    if len(years_sorted) < n_folds:
        raise ValueError(
            f"Cannot create {n_folds} folds from only {len(years_sorted)} unique years."
        )
    folds = [list(map(int, arr.tolist())) for arr in np.array_split(np.array(years_sorted), n_folds)]
    return [fold for fold in folds if fold]


def resolve_standard_splits(args: argparse.Namespace, available_years: Sequence[int]) -> Tuple[List[int], List[int], List[int]]:
    default_train_years = [y for y in range(1980, 1984) if y in available_years]
    train_years = parse_years(args.train_years, default_train_years)

    if not train_years:
        raise RuntimeError("No training years are available after resolving the requested range.")

    if args.val_years is None:
        candidate_val_years = [y for y in available_years if y > max(train_years)]
        val_years = candidate_val_years[:1]
    else:
        val_years = parse_years(args.val_years, [])

    if args.test_years is None:
        candidate_test_years = [
            y for y in available_years if y not in set(train_years).union(val_years)
        ]
        test_years = candidate_test_years[:1]
    else:
        test_years = parse_years(args.test_years, [])

    return train_years, val_years, test_years


def resolve_cv_years(args: argparse.Namespace, available_years: Sequence[int]) -> List[int]:
    """
    Resolve the year pool used for year-based cross-validation.

    Priority:
      1. Explicitly supplied --train-years / --val-years / --test-years union.
      2. Otherwise, all available years.
    """
    explicit_years: List[int] = []
    for values in (args.train_years, args.val_years, args.test_years):
        if values:
            explicit_years.extend(int(v) for v in values)

    if explicit_years:
        cv_years = sorted(set(y for y in explicit_years if y in available_years))
    else:
        cv_years = list(sorted(set(int(y) for y in available_years)))

    if not cv_years:
        raise RuntimeError("No years resolved for cross-validation.")

    return cv_years


def extract_year_from_path(path: Path) -> int:
    match = re.search(r"(\d{4})", path.name)
    if not match:
        raise ValueError(f"Could not extract year from filename: {path}")
    return int(match.group(1))


def build_file_map(pattern: str) -> Dict[int, Path]:
    paths = sorted(Path().glob(pattern))
    out: Dict[int, Path] = {}
    for p in paths:
        year = extract_year_from_path(p)
        out[year] = p
    return out


def infer_pressure_variable(ds: xr.Dataset, time_name: str, lat_name: str, lon_name: str) -> str:
    """
    Infer pressure variable as the first data variable with dimensions including
    time/lat/lon. Falls back to the first data variable if necessary.
    """
    for name, da in ds.data_vars.items():
        dims = set(da.dims)
        if time_name in dims and lat_name in dims and lon_name in dims:
            return name
    return next(iter(ds.data_vars))


# ---------------------------------------------------------------------
# Dataset indexing
# ---------------------------------------------------------------------

def collect_samples_for_years(
    years: Sequence[int],
    prmsl_files: Dict[int, Path],
    label_files: Dict[int, Path],
    hurricane_months: Sequence[int],
    time_name: str,
) -> List[Tuple[int, int]]:
    """
    Return a list of (year, time_index) samples filtered by hurricane months.
    """
    samples: List[Tuple[int, int]] = []

    for year in years:
        if year not in prmsl_files:
            print(f"[WARN] Missing PRMSL file for {year}; skipping.")
            continue
        if year not in label_files:
            print(f"[WARN] Missing LABELS file for {year}; skipping.")
            continue

        with xr.open_dataset(prmsl_files[year]) as ds:
            times = pd.to_datetime(ds[time_name].values)
            mask = np.array([t.month in hurricane_months for t in times], dtype=bool)
            indices = np.where(mask)[0].tolist()
            samples.extend((year, int(i)) for i in indices)

    return samples


class HurricaneHeatmapDataset(Dataset):
    """
    Lazy-loading dataset for PRMSL and Gaussian hurricane labels.

    Each item returns:
        x: torch.float32, shape (1, H, W)
        y: torch.float32, shape (1, H, W)
        metadata: dict with year, time_index, timestamp
    """

    def __init__(
        self,
        samples: Sequence[Tuple[int, int]],
        prmsl_files: Dict[int, Path],
        label_files: Dict[int, Path],
        pressure_var: Optional[str] = None,
        label_var: str = "tc_label",
        time_name: str = "time",
        lat_name: str = "lat",
        lon_name: str = "lon",
        normalize: str = "zscore_per_year",
        cache_open_datasets: bool = True,
    ) -> None:
        self.samples = list(samples)
        self.prmsl_files = prmsl_files
        self.label_files = label_files
        self.pressure_var = pressure_var
        self.label_var = label_var
        self.time_name = time_name
        self.lat_name = lat_name
        self.lon_name = lon_name
        self.normalize = normalize
        self.cache_open_datasets = cache_open_datasets

        self._pressure_ds_cache: Dict[int, xr.Dataset] = {}
        self._label_ds_cache: Dict[int, xr.Dataset] = {}
        self._pressure_var_by_year: Dict[int, str] = {}
        self._norm_stats_by_year: Dict[int, Tuple[float, float]] = {}

        self._initialise_year_metadata()

    def _open_pressure_ds(self, year: int) -> xr.Dataset:
        if self.cache_open_datasets:
            if year not in self._pressure_ds_cache:
                self._pressure_ds_cache[year] = xr.open_dataset(self.prmsl_files[year])
            return self._pressure_ds_cache[year]
        return xr.open_dataset(self.prmsl_files[year])

    def _open_label_ds(self, year: int) -> xr.Dataset:
        if self.cache_open_datasets:
            if year not in self._label_ds_cache:
                self._label_ds_cache[year] = xr.open_dataset(self.label_files[year])
            return self._label_ds_cache[year]
        return xr.open_dataset(self.label_files[year])

    def _initialise_year_metadata(self) -> None:
        years = sorted(set(y for y, _ in self.samples))
        for year in years:
            ds = xr.open_dataset(self.prmsl_files[year])
            var = self.pressure_var or infer_pressure_variable(
                ds, self.time_name, self.lat_name, self.lon_name
            )
            self._pressure_var_by_year[year] = var

            if self.normalize == "zscore_per_year":
                arr = ds[var].values.astype(np.float32)
                mean = float(np.nanmean(arr))
                std = float(np.nanstd(arr))
                if not np.isfinite(std) or std == 0:
                    std = 1.0
                self._norm_stats_by_year[year] = (mean, std)

            ds.close()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        year, time_index = self.samples[idx]

        pds = self._open_pressure_ds(year)
        lds = self._open_label_ds(year)

        pvar = self._pressure_var_by_year[year]

        x = pds[pvar].isel({self.time_name: time_index}).values.astype(np.float32)
        y = lds[self.label_var].isel({self.time_name: time_index}).values.astype(np.float32)

        # Labels are expected to be Gaussian amplitudes in [0, 1].
        y = np.clip(y, 0.0, 1.0)

        if self.normalize == "zscore_per_year":
            mean, std = self._norm_stats_by_year[year]
            x = (x - mean) / std
        elif self.normalize == "zscore_per_sample":
            mean = float(np.nanmean(x))
            std = float(np.nanstd(x))
            if not np.isfinite(std) or std == 0:
                std = 1.0
            x = (x - mean) / std
        elif self.normalize == "none":
            pass
        else:
            raise ValueError(f"Unknown normalization: {self.normalize}")

        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        x_tensor = torch.from_numpy(x[None, :, :]).float()
        y_tensor = torch.from_numpy(y[None, :, :]).float()

        timestamp = pd.to_datetime(pds[self.time_name].values[time_index])

        lat = np.asarray(pds[self.lat_name].values, dtype=np.float32)
        lon = np.asarray(pds[self.lon_name].values, dtype=np.float32)

        metadata = {
            "year": int(year),
            "time_index": int(time_index),
            "timestamp": str(timestamp),
            "lat": lat,
            "lon": lon
        }

        if not self.cache_open_datasets:
            pds.close()
            lds.close()

        return x_tensor, y_tensor, metadata

    def close(self) -> None:
        for ds in self._pressure_ds_cache.values():
            ds.close()
        for ds in self._label_ds_cache.values():
            ds.close()
        self._pressure_ds_cache.clear()
        self._label_ds_cache.clear()


# ---------------------------------------------------------------------
# U-Net model
# ---------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers += [
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet2D(nn.Module):
    """
    Small 2D U-Net for heatmap prediction.

    Output is raw logits. Apply sigmoid externally for probabilities/heatmaps.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, base_channels, dropout)
        self.enc2 = DoubleConv(base_channels, base_channels * 2, dropout)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4, dropout)
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8, dropout)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16, dropout)

        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_channels * 16, base_channels * 8, dropout)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4, dropout)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2, dropout)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels, dropout)

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    @staticmethod
    def _pad_or_crop_to_match(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        Pad or centre-crop x so that its spatial dimensions match ref.
        """
        target_h, target_w = ref.size(2), ref.size(3)
        h, w = x.size(2), x.size(3)

        # Crop if x is too large.
        if h > target_h:
            crop_top = (h - target_h) // 2
            x = x[:, :, crop_top:crop_top + target_h, :]
        if w > target_w:
            crop_left = (w - target_w) // 2
            x = x[:, :, :, crop_left:crop_left + target_w]

        # Pad if x is too small.
        h, w = x.size(2), x.size(3)
        pad_h = target_h - h
        pad_w = target_w - w

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x,
                [
                    pad_w // 2,
                    pad_w - pad_w // 2,
                    pad_h // 2,
                    pad_h - pad_h // 2,
                ],
            )

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        u4 = self.up4(b)
        u4 = self._pad_or_crop_to_match(u4, e4)
        d4 = self.dec4(torch.cat([u4, e4], dim=1))

        u3 = self.up3(d4)
        u3 = self._pad_or_crop_to_match(u3, e3)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(d3)
        u2 = self._pad_or_crop_to_match(u2, e2)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        u1 = self._pad_or_crop_to_match(u1, e1)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        return self.out_conv(d1)


# ---------------------------------------------------------------------
# Losses and metrics
# ---------------------------------------------------------------------

def weighted_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: float = 20.0,
) -> torch.Tensor:
    """
    Weighted BCE for sparse Gaussian labels.

    BCE handles soft labels in [0, 1], so Gaussian amplitudes are valid targets.
    """
    pos_weight_tensor = torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight_tensor,
    )


def mse_heatmap_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    preds = torch.sigmoid(logits)
    return F.mse_loss(preds, targets)


def dice_soft(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
    """
    Soft Dice on heatmaps, primarily diagnostic.
    """
    preds = preds.detach()
    targets = targets.detach()
    intersection = torch.sum(preds * targets)
    denominator = torch.sum(preds) + torch.sum(targets)
    return float((2.0 * intersection + eps) / (denominator + eps))


def peak_distance_diagnostics(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.4,
) -> Dict[str, float]:
    """
    Simple frame-level diagnostic:
        - number of predicted peaks
        - number of target peaks
        - nearest-neighbour distance in grid cells from each target peak to predicted peak

    This is intentionally simple and should be improved for publication-quality scoring.
    """
    pred_peaks = find_local_maxima(pred, threshold=threshold, min_distance=3)
    targ_peaks = find_local_maxima(target, threshold=threshold, min_distance=3)

    if len(targ_peaks) == 0 and len(pred_peaks) == 0:
        return {
            "n_pred": 0,
            "n_true": 0,
            "mean_nn_grid_distance": np.nan,
            "matched_targets": 0,
        }

    if len(targ_peaks) == 0:
        return {
            "n_pred": len(pred_peaks),
            "n_true": 0,
            "mean_nn_grid_distance": np.nan,
            "matched_targets": 0,
        }

    if len(pred_peaks) == 0:
        return {
            "n_pred": 0,
            "n_true": len(targ_peaks),
            "mean_nn_grid_distance": np.inf,
            "matched_targets": 0,
        }

    distances = []
    matched = 0

    for ty, tx, _ in targ_peaks:
        dmin = min(math.hypot(float(ty - py), float(tx - px)) for py, px, _ in pred_peaks)
        distances.append(dmin)
        matched += 1

    return {
        "n_pred": len(pred_peaks),
        "n_true": len(targ_peaks),
        "mean_nn_grid_distance": float(np.mean(distances)),
        "matched_targets": matched,
    }


# ---------------------------------------------------------------------
# Peak extraction and simple stitching
# ---------------------------------------------------------------------

def find_local_maxima(
    heatmap: np.ndarray,
    threshold: float = 0.4,
    min_distance: int = 3,
) -> List[Tuple[int, int, float]]:
    """
    Find local maxima in a 2D heatmap using max pooling logic.

    Returns:
        List of (iy, ix, score)
    """
    if heatmap.ndim != 2:
        raise ValueError("heatmap must be 2D")

    tensor = torch.from_numpy(heatmap[None, None, :, :].astype(np.float32))
    pooled = F.max_pool2d(
        tensor,
        kernel_size=2 * min_distance + 1,
        stride=1,
        padding=min_distance,
    )
    maxima = (tensor == pooled) & (tensor >= threshold)

    ys, xs = torch.where(maxima[0, 0])
    peaks: List[Tuple[int, int, float]] = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        peaks.append((int(y), int(x), float(heatmap[y, x])))

    peaks.sort(key=lambda p: p[2], reverse=True)
    return peaks


def stitch_detections_simple(
    detections: List[Dict],
    max_grid_distance_per_6h: float = 6.0,
    max_gap_steps: int = 2,
) -> List[List[Dict]]:
    """
    Very simple greedy track stitching.

    Each detection dict should contain at least:
        {
            "timestamp": pd.Timestamp,
            "step": int,
            "iy": int,
            "ix": int,
            "score": float,
        }

    This is a proof-of-concept tracker. For serious use, replace with Hungarian
    assignment or a Kalman-filter assignment scheme.
    """
    detections = sorted(detections, key=lambda d: d["timestamp"])
    tracks: List[List[Dict]] = []

    for det in detections:
        best_track_idx = None
        best_dist = float("inf")

        for ti, track in enumerate(tracks):
            last = track[-1]
            gap = det["step"] - last["step"]

            if gap <= 0 or gap > max_gap_steps:
                continue

            allowed = max_grid_distance_per_6h * gap
            dist = math.hypot(det["iy"] - last["iy"], det["ix"] - last["ix"])

            if dist <= allowed and dist < best_dist:
                best_dist = dist
                best_track_idx = ti

        if best_track_idx is None:
            tracks.append([det])
        else:
            tracks[best_track_idx].append(det)

    return tracks


# ---------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------

def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    loss_name: str,
    pos_weight: float,
) -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)

    losses: List[float] = []
    dices: List[float] = []

    for x, y, _meta in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)

        if loss_name == "bce":
            loss = weighted_bce_loss(logits, y, pos_weight=pos_weight)
        elif loss_name == "mse":
            loss = mse_heatmap_loss(logits, y)
        elif loss_name == "bce_mse":
            loss = weighted_bce_loss(logits, y, pos_weight=pos_weight) + 0.25 * mse_heatmap_loss(logits, y)
        else:
            raise ValueError(f"Unknown loss: {loss_name}")

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            preds = torch.sigmoid(logits)
            dice = dice_soft(preds, y)

        losses.append(float(loss.item()))
        dices.append(dice)

    return {
        "loss": float(np.mean(losses)) if losses else np.nan,
        "soft_dice": float(np.mean(dices)) if dices else np.nan,
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    threshold: float = 0.4,
    n_plot: int = 12,
    selected_time_indices: Optional[Sequence[int]] = None,
    selected_timestamps: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    plotted = 0

    selected_time_indices = (
        None if not selected_time_indices else set(int(x) for x in selected_time_indices)
    )
    selected_timestamps = (
        None if not selected_timestamps
        else set(str(pd.Timestamp(x)) for x in selected_timestamps)
    )


    for batch_idx, (x, y, meta) in enumerate(loader):
        x = x.to(device)
        logits = model(x)
        pred = torch.sigmoid(logits).cpu().numpy()
        targ = y.numpy()
        xin = x.cpu().numpy()

        batch_size = pred.shape[0]

        for i in range(batch_size):
            pred_i = pred[i, 0]
            targ_i = targ[i, 0]
            x_i = xin[i, 0]

            diag = peak_distance_diagnostics(pred_i, targ_i, threshold=threshold)

            year_val = meta["year"][i]
            if torch.is_tensor(year_val):
                year_val = int(year_val.item())
            else:
                year_val = int(year_val)

            time_index_val = meta["time_index"][i]
            if torch.is_tensor(time_index_val):
                time_index_val = int(time_index_val.item())
            else:
                time_index_val = int(time_index_val)

            row = {
                "batch": int(batch_idx),
                "item": int(i),
                "timestamp": str(meta["timestamp"][i]),
                "year": year_val,
                "time_index": time_index_val,
                **diag,
            }
            rows.append(row)

            timestamp_str = str(meta["timestamp"][i])

            plot_this = False
            if selected_time_indices is None and selected_timestamps is None:
                plot_this = plotted < n_plot
            else:
                if selected_time_indices is not None and time_index_val in selected_time_indices:
                    plot_this = True
                if selected_timestamps is not None and str(pd.Timestamp(timestamp_str)) in selected_timestamps:
                    plot_this = True

            if plot_this:
                fig, axes = plt.subplots(1, 3, figsize=(14, 4))

                lat_i = meta["lat"][i]
                lon_i = meta["lon"][i]

                if torch.is_tensor(lat_i):
                    lat = lat_i.cpu().numpy()
                else:
                    lat = np.asarray(lat_i)

                if torch.is_tensor(lon_i):
                    lon = lon_i.cpu().numpy()
                else:
                    lon = np.asarray(lon_i)

                origin = "lower" if lat[0] < lat[-1] else "upper"
                extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]

                im0 = axes[0].imshow(x_i, origin=origin, extent=extent, aspect="auto")
                axes[0].set_title("Input PRMSL, normalized")
                axes[0].set_xlabel("Longitude")
                axes[0].set_ylabel("Latitude")
                plt.colorbar(im0, ax=axes[0], fraction=0.046)

                im1 = axes[1].imshow(targ_i, origin=origin, extent=extent, vmin=0, vmax=1, aspect="auto")
                axes[1].set_title("Target Gaussian label")
                axes[1].set_xlabel("Longitude")
                axes[1].set_ylabel("Latitude")
                plt.colorbar(im1, ax=axes[1], fraction=0.046)

                im2 = axes[2].imshow(pred_i, origin=origin, extent=extent, vmin=0, vmax=1, aspect="auto")
                axes[2].set_title("Predicted heatmap")
                axes[2].set_xlabel("Longitude")
                axes[2].set_ylabel("Latitude")
                plt.colorbar(im2, ax=axes[2], fraction=0.046)

                fig.suptitle(timestamp_str)
                fig.tight_layout()

                safe_ts = pd.Timestamp(timestamp_str).strftime("%Y%m%d_%H%M%S")
                fig_path = out_dir / f"diagnostic_{safe_ts}_t{time_index_val:04d}.png"
                fig.savefig(fig_path, dpi=150)
                plt.close(fig)

                plotted += 1

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "evaluation_diagnostics.csv", index=False)
    return df


@torch.no_grad()
def run_detection_export(
    model: nn.Module,
    dataset: HurricaneHeatmapDataset,
    device: torch.device,
    out_csv: Path,
    threshold: float = 0.4,
    min_distance: int = 3,
) -> pd.DataFrame:
    """
    Export peak detections for every sample in a dataset.
    """
    model.eval()
    rows = []

    for idx in range(len(dataset)):
        x, _y, meta = dataset[idx]
        x_batch = x[None, :, :, :].to(device)

        pred = torch.sigmoid(model(x_batch))[0, 0].cpu().numpy()
        peaks = find_local_maxima(pred, threshold=threshold, min_distance=min_distance)
        timestamp = pd.Timestamp(meta["timestamp"])

        for iy, ix, score in peaks:
            rows.append({
                "dataset_index": int(idx),
                "step": int(idx),
                "year": int(meta["year"]),
                "time_index": int(meta["time_index"]),
                "timestamp": str(timestamp),
                "iy": int(iy),
                "ix": int(ix),
                "score": float(score),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df



def train_single_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_ds: Dataset,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    out_dir: Path,
    config: Dict,
) -> Tuple[List[Dict[str, float]], Path]:
    history: List[Dict[str, float]] = []
    best_val_loss = float("inf")
    best_model_path = out_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss_name=args.loss,
            pos_weight=args.pos_weight,
        )

        if len(val_ds) > 0:
            val_metrics = run_one_epoch(
                model,
                val_loader,
                optimizer=None,
                device=device,
                loss_name=args.loss,
                pos_weight=args.pos_weight,
            )
        else:
            val_metrics = {"loss": np.nan, "soft_dice": np.nan}

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_soft_dice": train_metrics["soft_dice"],
            "val_loss": val_metrics["loss"],
            "val_soft_dice": val_metrics["soft_dice"],
        }
        history.append(row)

        train_loss_str = f"{row['train_loss']:.6f}" if np.isfinite(row['train_loss']) else "nan"
        train_dice_str = f"{row['train_soft_dice']:.4f}" if np.isfinite(row['train_soft_dice']) else "nan"
        val_loss_str = f"{row['val_loss']:.6f}" if np.isfinite(row['val_loss']) else "nan"
        val_dice_str = f"{row['val_soft_dice']:.4f}" if np.isfinite(row['val_soft_dice']) else "nan"

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss_str}, "
            f"train_dice={train_dice_str}, "
            f"val_loss={val_loss_str}, "
            f"val_dice={val_dice_str}"
        )

        current_val_loss = row["val_loss"]
        if np.isfinite(current_val_loss) and current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                },
                best_model_path,
            )

    history_df = pd.DataFrame(history)
    history_df.to_csv(out_dir / "training_history.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="train")
    if history_df["val_loss"].notna().any():
        axes[0].plot(history_df["epoch"], history_df["val_loss"], label="validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training loss")
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["train_soft_dice"], label="train")
    if history_df["val_soft_dice"].notna().any():
        axes[1].plot(history_df["epoch"], history_df["val_soft_dice"], label="validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Soft Dice")
    axes[1].set_title("Soft Dice diagnostic")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close(fig)

    return history, best_model_path


def load_best_or_final_model(
    model: nn.Module,
    device: torch.device,
    out_dir: Path,
    config: Dict,
    epoch_hint: int,
) -> Path:
    best_model_path = out_dir / "best_model.pt"
    final_model_path = out_dir / "final_model.pt"

    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"Loaded best model from epoch {checkpoint['epoch']} "
            f"with val_loss={checkpoint['val_loss']:.6f}"
        )
        return best_model_path

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": epoch_hint,
            "val_loss": np.nan,
        },
        final_model_path,
    )
    print(f"No best validation checkpoint found; saved final model to {final_model_path}")
    return final_model_path


def fold_outputs_exist(fold_out_dir: Path) -> bool:
    return (
        (fold_out_dir / "holdout_diagnostics.csv").exists()
        and (fold_out_dir / "peak_detections.csv").exists()
        and (
            (fold_out_dir / "best_model.pt").exists()
            or (fold_out_dir / "final_model.pt").exists()
        )
    )


def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_existing_fold_outputs(
    fold_out_dir: Path,
    fold_idx: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    diag_df = safe_read_csv(fold_out_dir / "holdout_diagnostics.csv")
    det_df = safe_read_csv(fold_out_dir / "peak_detections.csv")
    hist_df = safe_read_csv(fold_out_dir / "training_history.csv")

    if not diag_df.empty and "fold" not in diag_df.columns:
        diag_df.insert(0, "fold", fold_idx)
    if not diag_df.empty and "split" not in diag_df.columns:
        diag_df.insert(1, "split", "holdout")

    if not det_df.empty and "fold" not in det_df.columns:
        det_df.insert(0, "fold", fold_idx)
    if not det_df.empty and "split" not in det_df.columns:
        det_df.insert(1, "split", "holdout")

    summary: Dict[str, float] = {"fold": fold_idx, "resumed_from_disk": True}
    if not hist_df.empty:
        last_row = hist_df.iloc[-1]
        summary.update({
            "last_train_loss": float(last_row.get("train_loss", np.nan)),
            "last_val_loss": float(last_row.get("val_loss", np.nan)),
            "last_train_soft_dice": float(last_row.get("train_soft_dice", np.nan)),
            "last_val_soft_dice": float(last_row.get("val_soft_dice", np.nan)),
        })
    return diag_df, det_df, summary


def run_cross_validation(
    args: argparse.Namespace,
    prmsl_files: Dict[int, Path],
    label_files: Dict[int, Path],
    available_years: Sequence[int],
    out_dir: Path,
) -> None:
    cv_years = resolve_cv_years(args, available_years)
    folds = year_folds(cv_years, args.cv_folds)

    print("Available years:", list(available_years))
    print("Cross-validation years:", cv_years)
    print("Cross-validation folds:", folds)
    print("Hurricane-season months:", args.season_months)
    if args.resume_cv:
        print("Resume mode: enabled")

    config = vars(args).copy()
    config.update({
        "available_years": list(available_years),
        "cv_years": cv_years,
        "cv_fold_years": folds,
        "cv_mode": True,
    })
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    all_diag_frames: List[pd.DataFrame] = []
    all_detection_frames: List[pd.DataFrame] = []
    fold_summary_rows: List[Dict[str, float]] = []

    for fold_idx, holdout_years in enumerate(folds, start=1):
        train_years = [y for y in cv_years if y not in set(holdout_years)]
        if not train_years:
            raise RuntimeError(
                f"Fold {fold_idx} has no training years after excluding holdout years {holdout_years}."
            )

        fold_out_dir = out_dir / f"fold_{fold_idx:02d}"
        fold_out_dir.mkdir(parents=True, exist_ok=True)

        print("-" * 80)
        print(f"Fold {fold_idx}/{len(folds)}")
        print("Training years:", train_years)
        print("Holdout years:", holdout_years)

        train_samples = collect_samples_for_years(
            train_years, prmsl_files, label_files, args.season_months, args.time_name
        )
        holdout_samples = collect_samples_for_years(
            holdout_years, prmsl_files, label_files, args.season_months, args.time_name
        )

        if args.resume_cv and fold_outputs_exist(fold_out_dir):
            print(f"Skipping completed fold {fold_idx}; using existing outputs in {fold_out_dir}")

            diag_df, det_df, summary = load_existing_fold_outputs(fold_out_dir, fold_idx)

            if not diag_df.empty:
                all_diag_frames.append(diag_df)
            if not det_df.empty:
                all_detection_frames.append(det_df)

            summary.update({
                "n_train_years": len(train_years),
                "n_holdout_years": len(holdout_years),
                "n_train_samples": len(train_samples),
                "n_holdout_samples": len(holdout_samples),
            })
            fold_summary_rows.append(summary)
            continue

        print(f"Train samples: {len(train_samples)}")
        print(f"Holdout samples: {len(holdout_samples)}")

        train_ds = HurricaneHeatmapDataset(
            train_samples,
            prmsl_files,
            label_files,
            pressure_var=args.pressure_var,
            label_var=args.label_var,
            time_name=args.time_name,
            lat_name=args.lat_name,
            lon_name=args.lon_name,
            normalize=args.normalize,
        )
        holdout_ds = HurricaneHeatmapDataset(
            holdout_samples,
            prmsl_files,
            label_files,
            pressure_var=args.pressure_var,
            label_var=args.label_var,
            time_name=args.time_name,
            lat_name=args.lat_name,
            lon_name=args.lon_name,
            normalize=args.normalize,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        holdout_loader = DataLoader(
            holdout_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        model = UNet2D(
            in_channels=1,
            out_channels=1,
            base_channels=args.base_channels,
            dropout=args.dropout,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        fold_config = dict(config)
        fold_config.update({
            "fold_index": fold_idx,
            "fold_train_years": train_years,
            "fold_holdout_years": holdout_years,
        })

        if args.skip_training:
            ckpt_path = Path(args.checkpoint) if args.checkpoint else (fold_out_dir / "best_model.pt")
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found for fold {fold_idx}: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(
                f"Loaded checkpoint from {ckpt_path} "
                f"(epoch={checkpoint.get('epoch', 'NA')}, "
                f"val_loss={checkpoint.get('val_loss', 'NA')})"
            )
            fold_summary_rows.append({
                "fold": fold_idx,
                "n_train_years": len(train_years),
                "n_holdout_years": len(holdout_years),
                "n_train_samples": len(train_samples),
                "n_holdout_samples": len(holdout_samples),
                "last_val_loss": float(checkpoint.get("val_loss", np.nan)),
                "resumed_from_disk": False,
            })
        else:
            history, _ = train_single_model(
                model=model,
                train_loader=train_loader,
                val_loader=holdout_loader,
                val_ds=holdout_ds,
                optimizer=optimizer,
                device=device,
                args=args,
                out_dir=fold_out_dir,
                config=fold_config,
            )
            fold_history_df = pd.DataFrame(history)
            if not fold_history_df.empty:
                last_row = fold_history_df.iloc[-1]
                fold_summary_rows.append({
                    "fold": fold_idx,
                    "n_train_years": len(train_years),
                    "n_holdout_years": len(holdout_years),
                    "n_train_samples": len(train_samples),
                    "n_holdout_samples": len(holdout_samples),
                    "last_train_loss": float(last_row["train_loss"]),
                    "last_val_loss": float(last_row["val_loss"]),
                    "last_train_soft_dice": float(last_row["train_soft_dice"]),
                    "last_val_soft_dice": float(last_row["val_soft_dice"]),
                })

        load_best_or_final_model(
            model=model,
            device=device,
            out_dir=fold_out_dir,
            config=fold_config,
            epoch_hint=args.epochs,
        )

        if len(holdout_ds) == 0:
            print(f"No holdout samples for fold {fold_idx}; skipping diagnostics and detection export.")
        else:
            print(f"Evaluating holdout years for fold {fold_idx}...")
            diag_df = evaluate_model(
                model,
                holdout_loader,
                device,
                fold_out_dir / "holdout_diagnostics",
                threshold=args.threshold,
                n_plot=args.diag_n_plot,
                selected_time_indices=args.diag_time_indices,
                selected_timestamps=args.diag_timestamps,
            )
            diag_df.insert(0, "fold", fold_idx)
            diag_df.insert(1, "split", "holdout")
            diag_df.to_csv(fold_out_dir / "holdout_diagnostics.csv", index=False)
            all_diag_frames.append(diag_df)

            detections_df = run_detection_export(
                model,
                holdout_ds,
                device,
                out_csv=fold_out_dir / "peak_detections.csv",
                threshold=args.threshold,
                min_distance=args.min_peak_distance,
            )
            if not detections_df.empty:
                detections_df.insert(0, "fold", fold_idx)
                detections_df.insert(1, "split", "holdout")
            all_detection_frames.append(detections_df)

        train_ds.close()
        holdout_ds.close()

    if all_diag_frames:
        cv_diag_df = pd.concat(all_diag_frames, ignore_index=True)
    else:
        cv_diag_df = pd.DataFrame()
    cv_diag_df.to_csv(out_dir / "cv_oos_diagnostics.csv", index=False)

    if all_detection_frames:
        cv_detection_df = pd.concat(all_detection_frames, ignore_index=True)
    else:
        cv_detection_df = pd.DataFrame()
    cv_detection_df.to_csv(out_dir / "cv_oos_peak_detections.csv", index=False)

    fold_summary_df = pd.DataFrame(fold_summary_rows)
    fold_summary_df.to_csv(out_dir / "cv_fold_summary.csv", index=False)

    if not cv_diag_df.empty:
        covered_years = sorted(cv_diag_df["year"].dropna().astype(int).unique().tolist())
        print("Out-of-sample predictions generated for years:", covered_years)
    else:
        print("No cross-validation diagnostics were generated.")

    print(f"Cross-validation complete. Outputs written to: {out_dir.resolve()}")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a U-Net hurricane detector from PRMSL and Gaussian IBTrACS labels."
    )

    parser.add_argument("--prmsl-pattern", default="Data_M1/PRMSL/PRMSL.*_6hourly.nc")
    parser.add_argument("--label-pattern", default="IBTrACS/LABELS.*_6hourly.nc")

    parser.add_argument("--pressure-var", default=None)
    parser.add_argument("--label-var", default="tc_label")

    parser.add_argument("--time-name", default="time")
    parser.add_argument("--lat-name", default="lat")
    parser.add_argument("--lon-name", default="lon")

    parser.add_argument("--train-years", nargs="*", type=int, default=None)
    parser.add_argument("--val-years", nargs="*", type=int, default=None)
    parser.add_argument("--test-years", nargs="*", type=int, default=None)

    parser.add_argument("--season-months", nargs="*", type=int, default=[6, 7, 8, 9, 10, 11])

    parser.add_argument(
        "--normalize",
        choices=["zscore_per_year", "zscore_per_sample", "none"],
        default="zscore_per_year",
    )

    parser.add_argument("--loss", choices=["bce", "mse", "bce_mse"], default="bce")
    parser.add_argument("--pos-weight", type=float, default=20.0)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)

    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--min-peak-distance", type=int, default=3)

    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--checkpoint", default=None)

    parser.add_argument("--diag-split", choices=["val", "test"], default="test")
    parser.add_argument("--diag-n-plot", type=int, default=12)

    parser.add_argument("--diag-time-indices", nargs="*", type=int, default=None)
    parser.add_argument("--diag-timestamps", nargs="*", default=None)

    parser.add_argument(
        "--cv-folds",
        type=int,
        default=None,
        help=(
            "Optional number of year-based cross-validation folds. If omitted, "
            "the script uses the existing train/validation/test split. If set to 2 "
            "or more, the model is trained repeatedly with year holdouts and "
            "out-of-sample predictions are exported across the selected year range."
        ),
    )
    parser.add_argument(
        "--resume-cv",
        action="store_true",
        help="Skip already completed CV folds and rebuild aggregate outputs.",
    )

    parser.add_argument("--out-dir", default="unet_outputs")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prmsl_files = build_file_map(args.prmsl_pattern)
    label_files = build_file_map(args.label_pattern)

    available_years = sorted(set(prmsl_files).intersection(label_files))
    if not available_years:
        raise RuntimeError("No matching PRMSL/LABELS year files found.")

    if args.cv_folds is not None:
        if args.cv_folds < 2:
            raise ValueError("--cv-folds must be at least 2 when provided.")
        run_cross_validation(
            args=args,
            prmsl_files=prmsl_files,
            label_files=label_files,
            available_years=available_years,
            out_dir=out_dir,
        )
        return

    train_years, val_years, test_years = resolve_standard_splits(args, available_years)

    print("Available years:", available_years)
    print("Train years:", train_years)
    print("Validation years:", val_years)
    print("Test years:", test_years)
    print("Hurricane-season months:", args.season_months)

    config = vars(args).copy()
    config.update({
        "available_years": available_years,
        "resolved_train_years": train_years,
        "resolved_val_years": val_years,
        "resolved_test_years": test_years,
        "cv_mode": False,
    })
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    train_samples = collect_samples_for_years(
        train_years, prmsl_files, label_files, args.season_months, args.time_name
    )
    val_samples = collect_samples_for_years(
        val_years, prmsl_files, label_files, args.season_months, args.time_name
    )
    test_samples = collect_samples_for_years(
        test_years, prmsl_files, label_files, args.season_months, args.time_name
    )

    print(f"Train samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")

    train_ds = HurricaneHeatmapDataset(
        train_samples,
        prmsl_files,
        label_files,
        pressure_var=args.pressure_var,
        label_var=args.label_var,
        time_name=args.time_name,
        lat_name=args.lat_name,
        lon_name=args.lon_name,
        normalize=args.normalize,
    )

    val_ds = HurricaneHeatmapDataset(
        val_samples,
        prmsl_files,
        label_files,
        pressure_var=args.pressure_var,
        label_var=args.label_var,
        time_name=args.time_name,
        lat_name=args.lat_name,
        lon_name=args.lon_name,
        normalize=args.normalize,
    )

    test_ds = HurricaneHeatmapDataset(
        test_samples,
        prmsl_files,
        label_files,
        pressure_var=args.pressure_var,
        label_var=args.label_var,
        time_name=args.time_name,
        lat_name=args.lat_name,
        lon_name=args.lon_name,
        normalize=args.normalize,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = UNet2D(
        in_channels=1,
        out_channels=1,
        base_channels=args.base_channels,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.skip_training:
        ckpt_path = Path(args.checkpoint) if args.checkpoint else (out_dir / "best_model.pt")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"Loaded checkpoint from {ckpt_path} "
            f"(epoch={checkpoint.get('epoch', 'NA')}, "
            f"val_loss={checkpoint.get('val_loss', 'NA')})"
        )
    else:
        train_single_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            val_ds=val_ds,
            optimizer=optimizer,
            device=device,
            args=args,
            out_dir=out_dir,
            config=config,
        )

    load_best_or_final_model(
        model=model,
        device=device,
        out_dir=out_dir,
        config=config,
        epoch_hint=args.epochs,
    )

    # Diagnostics on selected split only.
    if args.diag_split == "val":
        if len(val_ds) == 0:
            print("No validation samples available; skipping diagnostics.")
        else:
            print("Evaluating validation set...")
            target_loader = val_loader
            target_out_dir = out_dir / "val_diagnostics"

            diag_df = evaluate_model(
                model,
                target_loader,
                device,
                target_out_dir,
                threshold=args.threshold,
                n_plot=args.diag_n_plot,
                selected_time_indices=args.diag_time_indices,
                selected_timestamps=args.diag_timestamps,
            )
            print(diag_df.head())

            with pd.option_context("display.max_columns", None, "display.width", 200):
                print(diag_df.describe(include="all"))

    else:  # test
        if len(test_ds) == 0:
            print("No test samples available; skipping diagnostics.")
        else:
            print("Evaluating test set...")
            target_loader = test_loader
            target_out_dir = out_dir / "test_diagnostics"

            diag_df = evaluate_model(
                model,
                target_loader,
                device,
                target_out_dir,
                threshold=args.threshold,
                n_plot=args.diag_n_plot,
                selected_time_indices=args.diag_time_indices,
                selected_timestamps=args.diag_timestamps,
            )
            print(diag_df.head())

            with pd.option_context("display.max_columns", None, "display.width", 200):
                print(diag_df.describe(include="all"))

            print("Exporting test detections...")
            run_detection_export(
                model,
                test_ds,
                device,
                out_csv=out_dir / "peak_detections.csv",
                threshold=args.threshold,
                min_distance=args.min_peak_distance,
            )

    train_ds.close()
    val_ds.close()
    test_ds.close()

    print(f"Done. Outputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
