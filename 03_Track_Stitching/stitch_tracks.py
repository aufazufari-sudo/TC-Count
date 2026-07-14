#!/usr/bin/env python3
"""
Track-tuned stitching of U-Net peak detections into hurricane tracks.

This updates the earlier point-tuned workflow by choosing the score threshold
based on stitched-track / seasonal-count behaviour rather than point-level
6-hourly peaks alone.

Key default behaviour
---------------------
1. Sweep score thresholds over [0.75, 0.95] in steps of 0.01.
2. For each threshold:
   - filter detections,
   - suppress same-time duplicate peaks,
   - stitch tracks,
   - apply the persistence filter only to predicted tracks, leaving IBTrACS tracks unfiltered
     (default: ignore predicted tracks shorter than 48 h),
   - compute yearly track metrics matching the patched evaluator style.
3. Select a threshold using yearly track-count skill (default: lowest mean
   absolute yearly count error), with optional leave-one-year-out
   cross-validation diagnostics to reduce overfitting.
4. Re-run the final selected threshold and export stitched tracks plus yearly
   metrics similar to patched_yearly_track_metrics.csv.

Expected inputs
---------------
Detections CSV columns (minimum):
    year, timestamp, iy, ix, score, step, time_index

Supporting files typically available when you run the script:
    Data_M1/PRMSL/PRMSL.*_6hourly.nc
    IBTrACS/ibtracs.NA.list.v04r01.HU.csv
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None


EARTH_RADIUS_KM = 6371.0
DEFAULT_IBTRACS_CSV = "IBTrACS/ibtracs.NA.list.v04r01.HU.csv"
DEFAULT_PRMSL_PATTERN = "Data_M1/PRMSL/PRMSL.*_6hourly.nc"


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c


def ensure_datetime(series_or_df, col: Optional[str] = None) -> pd.Series:
    if col is None:
        out = pd.to_datetime(series_or_df, errors="coerce")
    else:
        out = pd.to_datetime(series_or_df[col], errors="coerce")
    if out.isna().any():
        raise ValueError(f"Found {int(out.isna().sum())} unparsable timestamps.")
    return out.dt.floor("s")


def extract_year_from_path(path: Path) -> int:
    match = re.search(r"(\d{4})", path.name)
    if not match:
        raise ValueError(f"Could not extract year from filename: {path}")
    return int(match.group(1))


def normalize_longitudes_to_dataset(lons_deg_e: np.ndarray, ds_lons: np.ndarray) -> np.ndarray:
    lons = np.asarray(lons_deg_e, dtype=float).copy()
    ds_lons = np.asarray(ds_lons, dtype=float)
    ds_min = float(np.nanmin(ds_lons))
    ds_max = float(np.nanmax(ds_lons))
    if ds_min >= 0 and ds_max > 180:
        return np.mod(lons, 360.0)
    return ((lons + 180.0) % 360.0) - 180.0


# -----------------------------------------------------------------------------
# Grid mapping
# -----------------------------------------------------------------------------

def infer_coord_names(ds: xr.Dataset, lat_name: Optional[str], lon_name: Optional[str]) -> Tuple[str, str]:
    if lat_name and lon_name:
        return lat_name, lon_name

    lat_candidates = ["lat", "latitude", "grid_yt"]
    lon_candidates = ["lon", "longitude", "grid_xt"]

    if lat_name is None:
        for cand in lat_candidates:
            if cand in ds.coords:
                lat_name = cand
                break
    if lon_name is None:
        for cand in lon_candidates:
            if cand in ds.coords:
                lon_name = cand
                break

    if lat_name is None or lon_name is None:
        raise KeyError("Could not infer latitude/longitude coordinate names from the dataset.")
    return lat_name, lon_name


def load_grid_from_any_prmsl(prmsl_pattern: str, lat_name: Optional[str], lon_name: Optional[str]):
    paths = sorted(Path().glob(prmsl_pattern))
    if not paths:
        raise FileNotFoundError(f"No PRMSL files match pattern: {prmsl_pattern}")
    with xr.open_dataset(paths[0]) as ds:
        lat_name, lon_name = infer_coord_names(ds, lat_name, lon_name)
        lat_vals = ds[lat_name].values
        lon_vals = ds[lon_name].values
    return lat_vals, lon_vals, lat_name, lon_name


def add_lat_lon_to_detections(df: pd.DataFrame, lat_vals: np.ndarray, lon_vals: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    iy = out["iy"].to_numpy(dtype=int)
    ix = out["ix"].to_numpy(dtype=int)
    if lat_vals.ndim == 1 and lon_vals.ndim == 1:
        if len(iy) and (iy.max() >= len(lat_vals) or ix.max() >= len(lon_vals)):
            raise IndexError("Detection iy/ix exceed available grid dimensions.")
        out["lat"] = lat_vals[iy]
        out["lon"] = lon_vals[ix]
    elif lat_vals.ndim == 2 and lon_vals.ndim == 2:
        if len(iy) and (iy.max() >= lat_vals.shape[0] or ix.max() >= lat_vals.shape[1]):
            raise IndexError("Detection iy/ix exceed available 2D grid dimensions.")
        out["lat"] = lat_vals[iy, ix]
        out["lon"] = lon_vals[iy, ix]
    else:
        raise ValueError("Latitude/longitude coordinates must both be 1D or both be 2D.")
    return out


# -----------------------------------------------------------------------------
# IBTrACS loading
# -----------------------------------------------------------------------------

def load_ibtracs(
    ibtracs_csv: str,
    pred_points_or_years: pd.DataFrame,
    status_filter: Optional[str] = "HU",
    months: Optional[Sequence[int]] = None,
    time_col: str = "ISO_TIME",
    status_col: str = "USA_STATUS",
    lat_col: str = "USA_LAT",
    lon_col: str = "USA_LON",
    sid_col: str = "SID",
    season_col: str = "SEASON",
    name_col: str = "NAME",
) -> pd.DataFrame:
    df = pd.read_csv(ibtracs_csv, low_memory=False)
    required = [time_col, lat_col, lon_col]
    for c in [status_col, sid_col, season_col, name_col]:
        if c in df.columns:
            required.append(c)
    missing = [c for c in set(required) if c not in df.columns]
    if missing:
        raise KeyError(f"IBTrACS CSV missing columns: {missing}")

    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out[lat_col] = pd.to_numeric(out[lat_col], errors="coerce")
    out[lon_col] = pd.to_numeric(out[lon_col], errors="coerce")
    out = out.dropna(subset=[time_col, lat_col, lon_col]).copy()

    if status_filter and status_col in out.columns:
        out = out[out[status_col].astype(str).str.upper() == status_filter.upper()].copy()

    years = sorted(pd.Series(pred_points_or_years["year"].unique()).astype(int).tolist())
    if season_col in out.columns:
        out = out[out[season_col].isin(years)].copy()
    else:
        out = out[out[time_col].dt.year.isin(years)].copy()

    if months is not None:
        out = out[out[time_col].dt.month.isin(list(months))].copy()

    pred_lons = pred_points_or_years["lon"].to_numpy() if "lon" in pred_points_or_years.columns else np.array([0.0, 360.0])
    out[lon_col] = normalize_longitudes_to_dataset(out[lon_col].to_numpy(), pred_lons)

    if sid_col not in out.columns:
        if season_col in out.columns:
            out[sid_col] = out[season_col].astype(str) + "_UNKNOWN"
        else:
            out[sid_col] = out[time_col].dt.year.astype(str) + "_UNKNOWN"
    if season_col not in out.columns:
        out[season_col] = out[time_col].dt.year.astype(int)
    if name_col not in out.columns:
        out[name_col] = "UNKNOWN"

    out = out.rename(columns={
        time_col: "timestamp",
        lat_col: "lat",
        lon_col: "lon",
        sid_col: "sid",
        season_col: "season",
        name_col: "name",
    })
    out["timestamp"] = out["timestamp"].dt.floor("s")
    return out[["sid", "season", "name", "timestamp", "lat", "lon"]].sort_values(["season", "sid", "timestamp"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Same-time suppression
# -----------------------------------------------------------------------------

def suppress_same_time_duplicates(df: pd.DataFrame, merge_radius_km: float) -> pd.DataFrame:
    kept_groups = []
    for ts, grp in df.groupby("timestamp", sort=True):
        grp = grp.sort_values("score", ascending=False).reset_index(drop=True)
        used = np.zeros(len(grp), dtype=bool)
        keep_rows = []
        for i in range(len(grp)):
            if used[i]:
                continue
            keep_rows.append(i)
            dists = haversine_km(grp.loc[i, "lat"], grp.loc[i, "lon"], grp["lat"].to_numpy(), grp["lon"].to_numpy())
            used = used | (dists <= merge_radius_km)
        kept_groups.append(grp.iloc[keep_rows])
    if not kept_groups:
        return df.iloc[0:0].copy()
    return pd.concat(kept_groups, ignore_index=True)


# -----------------------------------------------------------------------------
# Stitching
# -----------------------------------------------------------------------------

@dataclass
class Track:
    track_id: int
    points: List[dict] = field(default_factory=list)

    @property
    def last(self):
        return self.points[-1]

    def append(self, point: dict) -> None:
        self.points.append(point)

    @property
    def start_time(self):
        return self.points[0]["timestamp"]

    @property
    def end_time(self):
        return self.points[-1]["timestamp"]


def hours_between(t1: pd.Timestamp, t2: pd.Timestamp) -> float:
    return abs((pd.Timestamp(t2) - pd.Timestamp(t1)).total_seconds()) / 3600.0


def stitch_tracks(
    detections: pd.DataFrame,
    max_speed_kmh: float = 65.0,
    max_link_km: float = 450.0,
    max_gap_steps: int = 1,
    base_time_hours: float = 6.0,
    min_points: int = 1,
    min_duration_hours: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    detections = detections.copy()
    detections["timestamp"] = pd.to_datetime(detections["timestamp"]).dt.floor("s")
    detections = detections.sort_values(["year", "timestamp", "score"], ascending=[True, True, False]).reset_index(drop=True)

    all_tracks: List[Track] = []
    next_track_id = 1

    for year, year_df in detections.groupby("year", sort=True):
        active: List[Track] = []
        times = sorted(year_df["timestamp"].unique())

        for t in times:
            current = year_df[year_df["timestamp"] == t].copy().reset_index(drop=True)

            still_active: List[Track] = []
            for tr in active:
                dt_hours = hours_between(tr.end_time, t)
                if dt_hours <= (max_gap_steps + 1) * base_time_hours:
                    still_active.append(tr)
                else:
                    all_tracks.append(tr)
            active = still_active

            if len(active) == 0:
                for _, row in current.iterrows():
                    tr = Track(track_id=next_track_id)
                    next_track_id += 1
                    tr.append(row.to_dict())
                    active.append(tr)
                continue

            n_tracks = len(active)
            n_dets = len(current)
            cost = np.full((n_tracks, n_dets), 1.0e9, dtype=float)
            for i, tr in enumerate(active):
                last = tr.last
                dt_hours = hours_between(last["timestamp"], t)
                if dt_hours <= 0 or dt_hours > (max_gap_steps + 1) * base_time_hours:
                    continue
                allowed_km = min(max_link_km, max_speed_kmh * dt_hours)
                dists = haversine_km(last["lat"], last["lon"], current["lat"].to_numpy(), current["lon"].to_numpy())
                feasible = dists <= allowed_km
                this_cost = dists - 50.0 * current["score"].to_numpy()
                cost[i, feasible] = this_cost[feasible]

            matched_tracks = set()
            matched_dets = set()
            if linear_sum_assignment is not None and np.isfinite(cost).any():
                row_ind, col_ind = linear_sum_assignment(cost)
                for i, j in zip(row_ind, col_ind):
                    if cost[i, j] < 1.0e8:
                        active[i].append(current.iloc[j].to_dict())
                        matched_tracks.add(i)
                        matched_dets.add(j)
            else:
                pairs = []
                for i in range(n_tracks):
                    for j in range(n_dets):
                        if cost[i, j] < 1.0e8:
                            pairs.append((cost[i, j], i, j))
                pairs.sort()
                for _, i, j in pairs:
                    if i in matched_tracks or j in matched_dets:
                        continue
                    active[i].append(current.iloc[j].to_dict())
                    matched_tracks.add(i)
                    matched_dets.add(j)

            for j in range(n_dets):
                if j not in matched_dets:
                    tr = Track(track_id=next_track_id)
                    next_track_id += 1
                    tr.append(current.iloc[j].to_dict())
                    active.append(tr)

        all_tracks.extend(active)

    kept_tracks: List[Track] = []
    for tr in all_tracks:
        duration_h = hours_between(tr.start_time, tr.end_time)
        if len(tr.points) >= min_points and duration_h >= min_duration_hours:
            kept_tracks.append(tr)

    points_rows = []
    summary_rows = []
    for tr in kept_tracks:
        pts = pd.DataFrame(tr.points).sort_values("timestamp").reset_index(drop=True)
        duration_h = hours_between(pts.loc[0, "timestamp"], pts.loc[len(pts) - 1, "timestamp"])
        summary_rows.append({
            "track_id": tr.track_id,
            "year": int(pts.loc[0, "year"]),
            "start_time": pd.Timestamp(pts.loc[0, "timestamp"]),
            "end_time": pd.Timestamp(pts.loc[len(pts) - 1, "timestamp"]),
            "n_points": int(len(pts)),
            "duration_hours": float(duration_h),
            "mean_score": float(pts["score"].mean()) if "score" in pts.columns else np.nan,
            "max_score": float(pts["score"].max()) if "score" in pts.columns else np.nan,
            "start_lat": float(pts.loc[0, "lat"]),
            "start_lon": float(pts.loc[0, "lon"]),
            "end_lat": float(pts.loc[len(pts) - 1, "lat"]),
            "end_lon": float(pts.loc[len(pts) - 1, "lon"]),
        })
        for rank, (_, row) in enumerate(pts.iterrows(), start=1):
            d = row.to_dict()
            d["track_id"] = tr.track_id
            d["track_point_index"] = rank
            d["timestamp"] = pd.Timestamp(d["timestamp"])
            points_rows.append(d)

    points_df = pd.DataFrame(points_rows)
    summary_df = pd.DataFrame(summary_rows)
    if not points_df.empty:
        points_df = points_df.sort_values(["year", "timestamp", "track_id", "track_point_index"]).reset_index(drop=True)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["year", "start_time", "track_id"]).reset_index(drop=True)
    return points_df, summary_df


# -----------------------------------------------------------------------------
# Evaluation helpers (mirrors patched strict evaluation philosophy)
# -----------------------------------------------------------------------------

def build_track_summary_from_points(
    points: pd.DataFrame,
    id_col: str,
    year_col: Optional[str],
    season_col_name: str,
    name_col_value: Optional[str],
) -> pd.DataFrame:
    rows = []
    for tid, grp in points.groupby(id_col):
        grp = grp.sort_values("timestamp")
        start = pd.Timestamp(grp["timestamp"].iloc[0])
        end = pd.Timestamp(grp["timestamp"].iloc[-1])
        duration_h = (end - start).total_seconds() / 3600.0
        row = {
            id_col: tid,
            season_col_name: int(grp[year_col].iloc[0]) if year_col is not None else int(grp[season_col_name].iloc[0]),
            "start_time": start,
            "end_time": end,
            "n_points": int(len(grp)),
            "duration_hours": float(duration_h),
            "start_lat": float(grp["lat"].iloc[0]),
            "start_lon": float(grp["lon"].iloc[0]),
            "end_lat": float(grp["lat"].iloc[-1]),
            "end_lon": float(grp["lon"].iloc[-1]),
        }
        if "score" in grp.columns:
            row["mean_score"] = float(grp["score"].mean()) if grp["score"].notna().any() else np.nan
            row["max_score"] = float(grp["score"].max()) if grp["score"].notna().any() else np.nan
        if name_col_value is not None:
            row["name"] = name_col_value
        elif "name" in grp.columns:
            row["name"] = str(grp["name"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def apply_persistence_filter(
    points: pd.DataFrame,
    summary: pd.DataFrame,
    id_col: str,
    min_track_duration_hours: float,
    min_track_points: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if summary.empty:
        empty_points = points.iloc[0:0].copy()
        empty_summary = summary.iloc[0:0].copy()
        return empty_points, empty_summary, empty_summary
    keep_mask = (summary["duration_hours"] >= float(min_track_duration_hours)) & (summary["n_points"] >= int(min_track_points))
    kept_summary = summary[keep_mask].copy().reset_index(drop=True)
    excluded_summary = summary[~keep_mask].copy().reset_index(drop=True)
    keep_ids = set(kept_summary[id_col].tolist())
    kept_points = points[points[id_col].isin(keep_ids)].copy().reset_index(drop=True)
    return kept_points, kept_summary, excluded_summary


def longest_consecutive_true_run(flags: Sequence[bool]) -> int:
    best = 0
    current = 0
    for flag in flags:
        if flag:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def pair_diagnostics(
    pred_track: pd.DataFrame,
    truth_track: pd.DataFrame,
    max_match_radius_km: float,
    base_time_hours: float = 6.0,
    overlap_denominator: str = "shorter",
) -> Dict[str, float]:
    pred_track = pred_track.sort_values("timestamp").copy()
    truth_track = truth_track.sort_values("timestamp").copy()

    pred_times = set(pred_track["timestamp"])
    truth_times = set(truth_track["timestamp"])
    common_times = sorted(pred_times.intersection(truth_times))

    matched_timestamps = []
    matched_distances = []
    time_flags = []
    truth_indexed = {t: g for t, g in truth_track.groupby("timestamp")}
    pred_indexed = {t: g for t, g in pred_track.groupby("timestamp")}

    for t in sorted(truth_track["timestamp"].unique()):
        if t not in common_times:
            time_flags.append(False)
            continue
        psub = pred_indexed[t]
        tsub = truth_indexed[t]
        pred_lat = psub["lat"].to_numpy()[:, None]
        pred_lon = psub["lon"].to_numpy()[:, None]
        truth_lat = tsub["lat"].to_numpy()[None, :]
        truth_lon = tsub["lon"].to_numpy()[None, :]
        dmat = haversine_km(pred_lat, pred_lon, truth_lat, truth_lon)
        mindist = float(np.min(dmat))
        is_match = mindist <= max_match_radius_km
        time_flags.append(is_match)
        if is_match:
            matched_timestamps.append(t)
            matched_distances.append(mindist)

    n_truth_points = int(len(truth_track))
    n_pred_points = int(len(pred_track))
    n_common_times = int(len(common_times))
    n_matched_points = int(len(matched_timestamps))

    if overlap_denominator == "truth":
        denom = n_truth_points
    elif overlap_denominator == "pred":
        denom = n_pred_points
    elif overlap_denominator == "shorter":
        denom = min(n_truth_points, n_pred_points)
    elif overlap_denominator == "longer":
        denom = max(n_truth_points, n_pred_points)
    else:
        raise ValueError(f"Unsupported overlap_denominator: {overlap_denominator}")

    overlap_fraction = (n_matched_points / denom) if denom > 0 else np.nan
    longest_run_steps = longest_consecutive_true_run(time_flags)
    continuous_overlap_hours = float(max(0, longest_run_steps - 1) * base_time_hours) if longest_run_steps > 0 else 0.0

    return {
        "n_pred_points": float(n_pred_points),
        "n_truth_points": float(n_truth_points),
        "n_common_times": float(n_common_times),
        "n_matched_points": float(n_matched_points),
        "overlap_fraction": float(overlap_fraction) if overlap_fraction == overlap_fraction else np.nan,
        "continuous_overlap_hours": float(continuous_overlap_hours),
        "mean_match_distance_km": float(np.mean(matched_distances)) if matched_distances else np.nan,
        "max_match_distance_km": float(np.max(matched_distances)) if matched_distances else np.nan,
    }


def build_candidate_pairs(
    pred_points: pd.DataFrame,
    truth_points: pd.DataFrame,
    max_match_radius_km: float,
    min_overlap_points: int,
    min_overlap_fraction: float,
    min_continuous_overlap_hours: float,
    base_time_hours: float,
    overlap_denominator: str,
) -> Tuple[List[int], List[str], np.ndarray, pd.DataFrame, Dict[str, dict]]:
    pred_track_ids = sorted(pred_points["track_id"].unique().tolist()) if not pred_points.empty else []
    truth_sids = sorted(truth_points["sid"].unique().tolist()) if not truth_points.empty else []
    score_matrix = np.zeros((len(pred_track_ids), len(truth_sids)), dtype=float)
    pair_rows = []
    truth_meta: Dict[str, dict] = {}

    pred_groups = {tid: g.sort_values("timestamp") for tid, g in pred_points.groupby("track_id")} if not pred_points.empty else {}
    truth_groups = {sid: g.sort_values("timestamp") for sid, g in truth_points.groupby("sid")} if not truth_points.empty else {}

    for sid, g in truth_groups.items():
        truth_meta[sid] = {
            "season": int(g["season"].iloc[0]),
            "name": str(g["name"].iloc[0]),
            "start_time": str(g["timestamp"].iloc[0]),
            "end_time": str(g["timestamp"].iloc[-1]),
            "n_points": int(len(g)),
        }

    for i, tid in enumerate(pred_track_ids):
        pg = pred_groups[tid]
        for j, sid in enumerate(truth_sids):
            tg = truth_groups[sid]
            diag = pair_diagnostics(
                pred_track=pg,
                truth_track=tg,
                max_match_radius_km=max_match_radius_km,
                base_time_hours=base_time_hours,
                overlap_denominator=overlap_denominator,
            )
            is_candidate = (
                diag["n_matched_points"] >= float(min_overlap_points)
                and diag["overlap_fraction"] >= float(min_overlap_fraction)
                and diag["continuous_overlap_hours"] >= float(min_continuous_overlap_hours)
            )
            if is_candidate:
                score_matrix[i, j] = diag["n_matched_points"]
            pair_rows.append({
                "track_id": int(tid),
                "sid": sid,
                "season": int(truth_meta[sid]["season"]),
                "name": str(truth_meta[sid]["name"]),
                "is_candidate": bool(is_candidate),
                **diag,
            })

    pair_df = pd.DataFrame(pair_rows)
    if not pair_df.empty:
        pair_df = pair_df.sort_values(["season", "sid", "track_id"]).reset_index(drop=True)
    return pred_track_ids, truth_sids, score_matrix, pair_df, truth_meta


def solve_assignment(pred_track_ids: List[int], truth_sids: List[str], score_matrix: np.ndarray) -> List[Tuple[int, str, float]]:
    if score_matrix.size == 0:
        return []
    valid = score_matrix > 0
    if not np.any(valid):
        return []

    if linear_sum_assignment is None:
        pairs = []
        for i in range(len(pred_track_ids)):
            for j in range(len(truth_sids)):
                if valid[i, j]:
                    pairs.append((score_matrix[i, j], i, j))
        pairs.sort(reverse=True)
        used_i = set()
        used_j = set()
        chosen = []
        for score, i, j in pairs:
            if i in used_i or j in used_j:
                continue
            used_i.add(i)
            used_j.add(j)
            chosen.append((pred_track_ids[i], truth_sids[j], float(score)))
        return chosen

    cost = np.where(valid, -score_matrix, 1.0e9)
    row_ind, col_ind = linear_sum_assignment(cost)
    chosen = []
    for i, j in zip(row_ind, col_ind):
        if cost[i, j] < 1.0e8:
            chosen.append((pred_track_ids[i], truth_sids[j], float(score_matrix[i, j])))
    return chosen


def compute_metrics(pred_summary: pd.DataFrame, truth_points: pd.DataFrame, matches_df: pd.DataFrame) -> Dict[str, float]:
    n_pred = int(len(pred_summary))
    n_truth = int(truth_points["sid"].nunique()) if not truth_points.empty else 0
    n_matched = int(len(matches_df)) if not matches_df.empty else 0
    hit_rate = n_matched / n_truth if n_truth > 0 else np.nan
    false_alarm_rate = (n_pred - n_matched) / n_pred if n_pred > 0 else np.nan
    miss_rate = (n_truth - n_matched) / n_truth if n_truth > 0 else np.nan
    precision = n_matched / n_pred if n_pred > 0 else np.nan
    return {
        "n_pred_tracks": float(n_pred),
        "n_truth_tracks": float(n_truth),
        "matched_tracks": float(n_matched),
        "hit_rate": float(hit_rate) if hit_rate == hit_rate else np.nan,
        "false_alarm_rate": float(false_alarm_rate) if false_alarm_rate == false_alarm_rate else np.nan,
        "miss_rate": float(miss_rate) if miss_rate == miss_rate else np.nan,
        "precision": float(precision) if precision == precision else np.nan,
    }


def compute_yearly_metrics(
    pred_points: pd.DataFrame,
    pred_summary: pd.DataFrame,
    truth_points: pd.DataFrame,
    max_match_radius_km: float,
    min_overlap_points: int,
    min_overlap_fraction: float,
    min_continuous_overlap_hours: float,
    base_time_hours: float,
    overlap_denominator: str,
) -> pd.DataFrame:
    pred_years = set(pred_summary["year"].astype(int).unique().tolist()) if not pred_summary.empty else set()
    truth_years = set(truth_points["season"].astype(int).unique().tolist()) if not truth_points.empty else set()
    years = sorted(pred_years.union(truth_years))
    rows = []
    for year in years:
        pp = pred_points[pred_points["year"] == year].copy() if not pred_points.empty else pred_points.copy()
        ps = pred_summary[pred_summary["year"] == year].copy() if not pred_summary.empty else pred_summary.copy()
        tt = truth_points[truth_points["season"] == year].copy() if not truth_points.empty else truth_points.copy()
        if len(ps) == 0 and len(tt) == 0:
            continue
        pred_track_ids, truth_sids, score_matrix, pair_df, truth_meta = build_candidate_pairs(
            pred_points=pp,
            truth_points=tt,
            max_match_radius_km=max_match_radius_km,
            min_overlap_points=min_overlap_points,
            min_overlap_fraction=min_overlap_fraction,
            min_continuous_overlap_hours=min_continuous_overlap_hours,
            base_time_hours=base_time_hours,
            overlap_denominator=overlap_denominator,
        )
        assignments = solve_assignment(pred_track_ids, truth_sids, score_matrix)
        match_df = pd.DataFrame(assignments, columns=["track_id", "sid", "match_score"]) if assignments else pd.DataFrame(columns=["track_id", "sid", "match_score"])
        metrics = compute_metrics(ps, tt, match_df)
        metrics["year"] = int(year)
        rows.append(metrics)
    if not rows:
        return pd.DataFrame(columns=["year", "n_pred_tracks", "n_truth_tracks", "matched_tracks", "hit_rate", "false_alarm_rate", "miss_rate", "precision"])
    out = pd.DataFrame(rows)
    return out[["year", "n_pred_tracks", "n_truth_tracks", "matched_tracks", "hit_rate", "false_alarm_rate", "miss_rate", "precision"]].sort_values("year").reset_index(drop=True)


# -----------------------------------------------------------------------------
# Threshold sweep / selection
# -----------------------------------------------------------------------------

def summarise_threshold_from_yearly(yearly_df: pd.DataFrame) -> Dict[str, float]:
    if yearly_df.empty:
        return {
            "n_years": 0.0,
            "count_mae": np.nan,
            "count_rmse": np.nan,
            "count_bias": np.nan,
            "abs_total_count_bias": np.nan,
            "matched_tracks_total": 0.0,
            "mean_hit_rate": np.nan,
            "mean_precision": np.nan,
        }
    diff = yearly_df["n_pred_tracks"].astype(float) - yearly_df["n_truth_tracks"].astype(float)
    return {
        "n_years": float(len(yearly_df)),
        "count_mae": float(np.mean(np.abs(diff))),
        "count_rmse": float(np.sqrt(np.mean(diff ** 2))),
        "count_bias": float(np.mean(diff)),
        "abs_total_count_bias": float(abs(np.sum(diff))),
        "matched_tracks_total": float(yearly_df["matched_tracks"].sum()),
        "mean_hit_rate": float(yearly_df["hit_rate"].mean()) if yearly_df["hit_rate"].notna().any() else np.nan,
        "mean_precision": float(yearly_df["precision"].mean()) if yearly_df["precision"].notna().any() else np.nan,
    }


def build_threshold_score_table(yearly_by_threshold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for thr, grp in yearly_by_threshold.groupby("score_threshold"):
        row = {"score_threshold": float(thr)}
        row.update(summarise_threshold_from_yearly(grp))
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=[
            "score_threshold", "n_years", "count_mae", "count_rmse", "count_bias",
            "abs_total_count_bias", "matched_tracks_total", "mean_hit_rate", "mean_precision"
        ])
    return pd.DataFrame(rows).sort_values("score_threshold").reset_index(drop=True)


def pick_best_threshold(summary_df: pd.DataFrame, criterion: str = "count_mae") -> float:
    if summary_df.empty:
        raise ValueError("Cannot pick threshold from an empty summary table.")

    df = summary_df.copy()
    ascending_map = {
        "count_mae": True,
        "count_rmse": True,
        "abs_total_count_bias": True,
        "matched_tracks_total": False,
        "mean_hit_rate": False,
        "mean_precision": False,
    }
    if criterion not in ascending_map:
        raise ValueError(f"Unsupported threshold criterion: {criterion}")

    # Deterministic tie-breaks: favour lower count error, then better matching,
    # then slightly higher threshold if still tied (to curb false alarms).
    sort_cols = [criterion, "count_mae", "count_rmse", "abs_total_count_bias", "matched_tracks_total", "mean_hit_rate", "mean_precision", "score_threshold"]
    ascending = [ascending_map[criterion], True, True, True, False, False, False, False]
    best = df.sort_values(sort_cols, ascending=ascending).iloc[0]
    return float(best["score_threshold"])


def cross_validate_thresholds(yearly_by_threshold: pd.DataFrame, criterion: str = "count_mae") -> Tuple[pd.DataFrame, Optional[float]]:
    if yearly_by_threshold.empty:
        return pd.DataFrame(), None
    years = sorted(yearly_by_threshold["year"].unique().tolist())
    if len(years) < 3:
        return pd.DataFrame(), None

    fold_rows = []
    for test_year in years:
        train = yearly_by_threshold[yearly_by_threshold["year"] != test_year].copy()
        test = yearly_by_threshold[yearly_by_threshold["year"] == test_year].copy()
        train_summary = build_threshold_score_table(train)
        if train_summary.empty:
            continue
        chosen_thr = pick_best_threshold(train_summary, criterion=criterion)
        test_row = test[test["score_threshold"] == chosen_thr].copy()
        if test_row.empty:
            continue
        r = test_row.iloc[0].to_dict()
        count_error = float(r["n_pred_tracks"] - r["n_truth_tracks"])
        fold_rows.append({
            "test_year": int(test_year),
            "selected_threshold": float(chosen_thr),
            "train_count_mae": float(train_summary.loc[train_summary["score_threshold"] == chosen_thr, "count_mae"].iloc[0]),
            "train_count_rmse": float(train_summary.loc[train_summary["score_threshold"] == chosen_thr, "count_rmse"].iloc[0]),
            "holdout_n_pred_tracks": float(r["n_pred_tracks"]),
            "holdout_n_truth_tracks": float(r["n_truth_tracks"]),
            "holdout_matched_tracks": float(r["matched_tracks"]),
            "holdout_hit_rate": float(r["hit_rate"]) if pd.notna(r["hit_rate"]) else np.nan,
            "holdout_precision": float(r["precision"]) if pd.notna(r["precision"]) else np.nan,
            "holdout_count_error": float(count_error),
            "holdout_abs_count_error": float(abs(count_error)),
        })

    cv_df = pd.DataFrame(fold_rows)
    if cv_df.empty:
        return cv_df, None

    freq = Counter(np.round(cv_df["selected_threshold"].to_numpy(dtype=float), 6).tolist())
    max_freq = max(freq.values())
    modal_thresholds = sorted([thr for thr, cnt in freq.items() if cnt == max_freq])
    chosen = float(np.median(modal_thresholds))
    return cv_df.sort_values("test_year").reset_index(drop=True), chosen


# -----------------------------------------------------------------------------
# One-threshold full run
# -----------------------------------------------------------------------------

def evaluate_single_threshold(
    detections_with_latlon: pd.DataFrame,
    truth_points_raw: pd.DataFrame,
    threshold: float,
    same_time_merge_radius_km: float,
    max_speed_kmh: float,
    max_link_km: float,
    max_gap_steps: int,
    base_time_hours: float,
    stitch_min_points: int,
    stitch_min_duration_hours: float,
    min_track_duration_hours: float,
    min_track_points: int,
    max_match_radius_km: float,
    min_overlap_points: int,
    min_overlap_fraction: float,
    min_continuous_overlap_hours: float,
    overlap_denominator: str,
) -> Dict[str, object]:
    filtered = detections_with_latlon[detections_with_latlon["score"] >= float(threshold)].copy()
    filtered = suppress_same_time_duplicates(filtered, merge_radius_km=same_time_merge_radius_km)
    filtered = filtered.sort_values(["year", "timestamp", "score"], ascending=[True, True, False]).reset_index(drop=True)

    pred_points_raw, pred_summary_raw = stitch_tracks(
        detections=filtered,
        max_speed_kmh=max_speed_kmh,
        max_link_km=max_link_km,
        max_gap_steps=max_gap_steps,
        base_time_hours=base_time_hours,
        min_points=stitch_min_points,
        min_duration_hours=stitch_min_duration_hours,
    )

    if pred_summary_raw.empty:
        pred_summary_raw = pd.DataFrame(columns=["track_id", "year", "start_time", "end_time", "n_points", "duration_hours"])
        pred_points_raw = pred_points_raw.iloc[0:0].copy() if not pred_points_raw.empty else pd.DataFrame(columns=["track_id", "year", "timestamp", "lat", "lon"])

    truth_summary_raw = build_track_summary_from_points(
        points=truth_points_raw,
        id_col="sid",
        year_col="season",
        season_col_name="season",
        name_col_value=None,
    ) if not truth_points_raw.empty else pd.DataFrame(columns=["sid", "season", "start_time", "end_time", "n_points", "duration_hours"])

    pred_points, pred_summary, pred_excluded = apply_persistence_filter(
        points=pred_points_raw,
        summary=pred_summary_raw,
        id_col="track_id",
        min_track_duration_hours=min_track_duration_hours,
        min_track_points=min_track_points,
    )

    # Leave truth/IBTrACS tracks unfiltered so evaluation uses the full reference set.
    truth_points = truth_points_raw.copy().reset_index(drop=True)
    truth_summary = truth_summary_raw.copy().reset_index(drop=True)
    truth_excluded = truth_summary_raw.iloc[0:0].copy()

    pred_track_ids, truth_sids, score_matrix, pair_df, truth_meta = build_candidate_pairs(
        pred_points=pred_points,
        truth_points=truth_points,
        max_match_radius_km=max_match_radius_km,
        min_overlap_points=min_overlap_points,
        min_overlap_fraction=min_overlap_fraction,
        min_continuous_overlap_hours=min_continuous_overlap_hours,
        base_time_hours=base_time_hours,
        overlap_denominator=overlap_denominator,
    )
    assignments = solve_assignment(pred_track_ids, truth_sids, score_matrix)

    match_rows = []
    for track_id, sid, match_score in assignments:
        meta = truth_meta[sid]
        pair_row = pair_df[(pair_df["track_id"] == track_id) & (pair_df["sid"] == sid)].iloc[0]
        match_rows.append({
            "track_id": int(track_id),
            "sid": sid,
            "season": int(meta["season"]),
            "name": str(meta["name"]),
            "match_score": float(match_score),
            "n_matched_points": int(pair_row["n_matched_points"]),
            "overlap_fraction": float(pair_row["overlap_fraction"]),
            "continuous_overlap_hours": float(pair_row["continuous_overlap_hours"]),
            "mean_match_distance_km": float(pair_row["mean_match_distance_km"]) if pd.notna(pair_row["mean_match_distance_km"]) else np.nan,
            "truth_start_time": meta["start_time"],
            "truth_end_time": meta["end_time"],
        })
    matches_df = pd.DataFrame(match_rows)
    if not matches_df.empty:
        matches_df = matches_df.sort_values(["season", "sid", "track_id"]).reset_index(drop=True)

    overall = compute_metrics(pred_summary, truth_points, matches_df)
    yearly = compute_yearly_metrics(
        pred_points=pred_points,
        pred_summary=pred_summary,
        truth_points=truth_points,
        max_match_radius_km=max_match_radius_km,
        min_overlap_points=min_overlap_points,
        min_overlap_fraction=min_overlap_fraction,
        min_continuous_overlap_hours=min_continuous_overlap_hours,
        base_time_hours=base_time_hours,
        overlap_denominator=overlap_denominator,
    )

    return {
        "threshold": float(threshold),
        "filtered_detections": filtered,
        "pred_points_raw": pred_points_raw,
        "pred_summary_raw": pred_summary_raw,
        "pred_points": pred_points,
        "pred_summary": pred_summary,
        "pred_excluded": pred_excluded,
        "truth_points": truth_points,
        "truth_summary": truth_summary,
        "truth_excluded": truth_excluded,
        "pair_df": pair_df,
        "matches_df": matches_df,
        "overall": overall,
        "yearly": yearly,
    }


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Track-tuned stitching of U-Net detections into hurricane tracks.")
    p.add_argument("--detections-csv", default="unet_outputs/peak_detections.csv", help="CSV of peak detections.")
    p.add_argument("--out-dir", default="track_stitch_track_tuned_outputs", help="Output directory.")

    p.add_argument("--prmsl-pattern", default=DEFAULT_PRMSL_PATTERN, help="Glob pattern for PRMSL files used to recover lat/lon grid.")
    p.add_argument("--lat-name", default=None, help="Latitude coordinate name, if not inferable.")
    p.add_argument("--lon-name", default=None, help="Longitude coordinate name, if not inferable.")

    p.add_argument("--ibtracs-csv", default=DEFAULT_IBTRACS_CSV, help="IBTrACS hurricane CSV.")
    p.add_argument("--ibtracs-status", default="HU", help="IBTrACS status filter (default: HU). Use ALL to disable.")
    p.add_argument("--months", nargs="*", type=int, default=[6, 7, 8, 9, 10, 11], help="Months to evaluate.")

    # Threshold selection
    p.add_argument("--score-threshold", type=float, default=None, help="Manual score threshold. If omitted, a sweep is performed.")
    p.add_argument("--threshold-grid", type=float, nargs="*", default=None, help="Explicit threshold grid, e.g. 0.75 0.80 0.85 0.90 0.95")
    p.add_argument("--threshold-min", type=float, default=0.75, help="Default minimum threshold for track-based sweep.")
    p.add_argument("--threshold-max", type=float, default=0.95, help="Default maximum threshold for track-based sweep.")
    p.add_argument("--threshold-step", type=float, default=0.01, help="Default threshold step for track-based sweep.")
    p.add_argument("--threshold-criterion", choices=["count_mae", "count_rmse", "abs_total_count_bias", "matched_tracks_total", "mean_hit_rate", "mean_precision"], default="count_mae", help="Criterion used to pick a threshold from yearly track metrics.")
    p.add_argument("--selection-mode", choices=["direct", "cv"], default="cv", help="Threshold selection mode. 'cv' uses leave-one-year-out diagnostics and chooses the modal fold-selected threshold when possible.")

    # Same-time suppression
    p.add_argument("--same-time-merge-radius-km", type=float, default=250.0, help="Within each timestamp, suppress duplicate peaks closer than this radius.")

    # Stitching dynamics
    p.add_argument("--max-speed-kmh", type=float, default=65.0, help="Maximum translation speed for linking nodes.")
    p.add_argument("--max-link-km", type=float, default=450.0, help="Hard cap on point-to-point link distance.")
    p.add_argument("--max-gap-steps", type=int, default=1, help="Maximum number of missing 6-hour steps allowed while keeping a track active.")
    p.add_argument("--base-time-hours", type=float, default=6.0, help="Nominal timestep in hours.")
    p.add_argument("--stitch-min-points", type=int, default=1, help="Internal post-stitch minimum points for predicted tracks before final evaluation filtering.")
    p.add_argument("--stitch-min-duration-hours", type=float, default=0.0, help="Internal post-stitch minimum duration for predicted tracks before final evaluation filtering.")

    # Predicted-track persistence filter
    p.add_argument("--min-track-duration-hours", type=float, default=48.0, help="Minimum duration applied only to predicted tracks before evaluation; truth/IBTrACS tracks are left unfiltered.")
    p.add_argument("--min-track-points", type=int, default=1, help="Minimum point count applied only to predicted tracks before evaluation; truth/IBTrACS tracks are left unfiltered.")

    # Strict track matching
    p.add_argument("--max-match-radius-km", type=float, default=300.0, help="Maximum pointwise distance for a matched timestamp.")
    p.add_argument("--min-overlap-points", type=int, default=2, help="Minimum matched timestamps for a candidate track pair.")
    p.add_argument("--min-overlap-fraction", type=float, default=0.10, help="Minimum overlap fraction for a candidate track pair.")
    p.add_argument("--min-continuous-overlap-hours", type=float, default=6.0, help="Minimum continuous overlap duration in hours for a candidate track pair.")
    p.add_argument("--overlap-denominator", choices=["truth", "pred", "shorter", "longer"], default="shorter", help="Denominator used for overlap fraction.")

    return p.parse_args()


def save_time_columns_as_str(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].astype(str)
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detections = pd.read_csv(args.detections_csv)
    required_cols = {"year", "timestamp", "iy", "ix", "score", "step", "time_index"}
    missing = sorted(required_cols.difference(detections.columns))
    if missing:
        raise KeyError(f"Detections CSV is missing columns: {missing}")
    detections = detections.copy()
    detections["timestamp"] = ensure_datetime(detections["timestamp"])

    lat_vals, lon_vals, lat_name, lon_name = load_grid_from_any_prmsl(args.prmsl_pattern, args.lat_name, args.lon_name)
    detections = add_lat_lon_to_detections(detections, lat_vals, lon_vals)
    detections = detections.sort_values(["year", "timestamp", "score"], ascending=[True, True, False]).reset_index(drop=True)

    ibtracs_path = Path(args.ibtracs_csv)
    if not ibtracs_path.exists():
        raise FileNotFoundError(f"IBTrACS CSV not found at: {ibtracs_path}")
    truth_points_raw = load_ibtracs(
        ibtracs_csv=str(ibtracs_path),
        pred_points_or_years=detections,
        status_filter=None if str(args.ibtracs_status).upper() == "ALL" else args.ibtracs_status,
        months=args.months,
    )

    if args.score_threshold is not None:
        thresholds = [float(args.score_threshold)]
    elif args.threshold_grid and len(args.threshold_grid) > 0:
        thresholds = sorted({round(float(x), 6) for x in args.threshold_grid})
    else:
        thresholds = list(np.round(np.arange(args.threshold_min, args.threshold_max + 0.5 * args.threshold_step, args.threshold_step), 6))
    if not thresholds:
        raise ValueError("Threshold list is empty.")

    sweep_yearly_rows = []
    sweep_summary_rows = []
    cached_results: Dict[float, Dict[str, object]] = {}

    for thr in thresholds:
        res = evaluate_single_threshold(
            detections_with_latlon=detections,
            truth_points_raw=truth_points_raw,
            threshold=float(thr),
            same_time_merge_radius_km=args.same_time_merge_radius_km,
            max_speed_kmh=args.max_speed_kmh,
            max_link_km=args.max_link_km,
            max_gap_steps=args.max_gap_steps,
            base_time_hours=args.base_time_hours,
            stitch_min_points=args.stitch_min_points,
            stitch_min_duration_hours=args.stitch_min_duration_hours,
            min_track_duration_hours=args.min_track_duration_hours,
            min_track_points=args.min_track_points,
            max_match_radius_km=args.max_match_radius_km,
            min_overlap_points=args.min_overlap_points,
            min_overlap_fraction=args.min_overlap_fraction,
            min_continuous_overlap_hours=args.min_continuous_overlap_hours,
            overlap_denominator=args.overlap_denominator,
        )
        cached_results[float(thr)] = res

        yearly = res["yearly"].copy()
        if not yearly.empty:
            yearly.insert(0, "score_threshold", float(thr))
            sweep_yearly_rows.append(yearly)

        summary_row = {
            "score_threshold": float(thr),
            "n_filtered_detections": float(len(res["filtered_detections"])),
            "n_pred_tracks_raw": float(len(res["pred_summary_raw"])),
            "n_pred_tracks_filtered": float(len(res["pred_summary"])),
            "n_truth_tracks_filtered": float(res["truth_points"]["sid"].nunique()) if not res["truth_points"].empty else 0.0,
        }
        summary_row.update(summarise_threshold_from_yearly(res["yearly"]))
        sweep_summary_rows.append(summary_row)

    sweep_yearly_df = pd.concat(sweep_yearly_rows, ignore_index=True) if sweep_yearly_rows else pd.DataFrame(columns=["score_threshold", "year", "n_pred_tracks", "n_truth_tracks", "matched_tracks", "hit_rate", "false_alarm_rate", "miss_rate", "precision"])
    sweep_summary_df = pd.DataFrame(sweep_summary_rows).sort_values("score_threshold").reset_index(drop=True)

    sweep_yearly_df.to_csv(out_dir / "threshold_sweep_yearly_track_metrics.csv", index=False)
    sweep_summary_df.to_csv(out_dir / "threshold_sweep_summary.csv", index=False)

    cv_df, cv_threshold = cross_validate_thresholds(sweep_yearly_df, criterion=args.threshold_criterion)
    if not cv_df.empty:
        cv_df.to_csv(out_dir / "threshold_cross_validation_by_year.csv", index=False)

    direct_threshold = pick_best_threshold(sweep_summary_df[[
        "score_threshold", "count_mae", "count_rmse", "abs_total_count_bias", "matched_tracks_total", "mean_hit_rate", "mean_precision"
    ]], criterion=args.threshold_criterion)

    if len(thresholds) == 1:
        final_threshold = float(thresholds[0])
        threshold_selection_reason = "manual_or_single_threshold"
    elif args.selection_mode == "cv" and cv_threshold is not None:
        final_threshold = float(cv_threshold)
        threshold_selection_reason = "leave_one_year_out_modal_threshold"
    else:
        final_threshold = float(direct_threshold)
        threshold_selection_reason = "direct_full_sample_threshold"

    final_result = cached_results[final_threshold]

    filtered_detections = final_result["filtered_detections"]
    pred_points_raw = final_result["pred_points_raw"]
    pred_summary_raw = final_result["pred_summary_raw"]
    pred_points = final_result["pred_points"]
    pred_summary = final_result["pred_summary"]
    pred_excluded = final_result["pred_excluded"]
    truth_points = final_result["truth_points"]
    truth_summary = final_result["truth_summary"]
    truth_excluded = final_result["truth_excluded"]
    pair_df = final_result["pair_df"]
    matches_df = final_result["matches_df"]
    overall = final_result["overall"]
    yearly = final_result["yearly"]

    filtered_detections_out = filtered_detections.copy()
    if "timestamp" in filtered_detections_out.columns:
        filtered_detections_out["timestamp"] = filtered_detections_out["timestamp"].astype(str)
    filtered_detections_out.to_csv(out_dir / "filtered_detections.csv", index=False)

    pred_points_raw_out = pred_points_raw.copy()
    if "timestamp" in pred_points_raw_out.columns:
        pred_points_raw_out["timestamp"] = pred_points_raw_out["timestamp"].astype(str)
    pred_points_raw_out.to_csv(out_dir / "stitched_track_points_raw.csv", index=False)
    save_time_columns_as_str(pred_summary_raw, ["start_time", "end_time"]).to_csv(out_dir / "stitched_track_summary_raw.csv", index=False)

    pred_points_out = pred_points.copy()
    if "timestamp" in pred_points_out.columns:
        pred_points_out["timestamp"] = pred_points_out["timestamp"].astype(str)
    pred_points_out.to_csv(out_dir / "stitched_track_points.csv", index=False)
    save_time_columns_as_str(pred_summary, ["start_time", "end_time"]).to_csv(out_dir / "stitched_track_summary.csv", index=False)

    save_time_columns_as_str(pred_summary, ["start_time", "end_time"]).to_csv(out_dir / "filtered_predicted_track_summary.csv", index=False)
    save_time_columns_as_str(truth_summary, ["start_time", "end_time"]).to_csv(out_dir / "filtered_ibtracs_track_summary.csv", index=False)
    save_time_columns_as_str(pred_excluded, ["start_time", "end_time"]).to_csv(out_dir / "excluded_predicted_tracks_by_persistence.csv", index=False)
    save_time_columns_as_str(truth_excluded, ["start_time", "end_time"]).to_csv(out_dir / "excluded_ibtracs_tracks_by_persistence.csv", index=False)

    pair_df.to_csv(out_dir / "patched_candidate_pair_diagnostics.csv", index=False)
    matches_df.to_csv(out_dir / "patched_predicted_to_ibtracs_matches.csv", index=False)
    yearly.to_csv(out_dir / "patched_yearly_track_metrics.csv", index=False)
    with open(out_dir / "patched_overall_track_metrics.json", "w") as f:
        json.dump(overall, f, indent=2)

    matched_pred = set(matches_df["track_id"].tolist()) if not matches_df.empty else set()
    matched_truth = set(matches_df["sid"].tolist()) if not matches_df.empty else set()
    unmatched_pred = pred_summary[~pred_summary["track_id"].isin(matched_pred)].copy()
    unmatched_truth = truth_summary[~truth_summary["sid"].isin(matched_truth)].copy()
    save_time_columns_as_str(unmatched_pred, ["start_time", "end_time"]).to_csv(out_dir / "patched_unmatched_predicted_tracks.csv", index=False)
    save_time_columns_as_str(unmatched_truth, ["start_time", "end_time"]).to_csv(out_dir / "patched_unmatched_ibtracs_tracks.csv", index=False)

    run_summary = {
        "detections_csv": str(Path(args.detections_csv).resolve()),
        "ibtracs_csv": str(Path(args.ibtracs_csv).resolve()),
        "months": list(args.months),
        "thresholds": [float(x) for x in thresholds],
        "manual_threshold": float(args.score_threshold) if args.score_threshold is not None else None,
        "selection_mode": str(args.selection_mode),
        "threshold_criterion": str(args.threshold_criterion),
        "direct_best_threshold": float(direct_threshold),
        "cv_modal_threshold": float(cv_threshold) if cv_threshold is not None else None,
        "final_threshold": float(final_threshold),
        "threshold_selection_reason": threshold_selection_reason,
        "min_track_duration_hours": float(args.min_track_duration_hours),
        "min_track_points": int(args.min_track_points),
        "stitch_min_points": int(args.stitch_min_points),
        "stitch_min_duration_hours": float(args.stitch_min_duration_hours),
        "same_time_merge_radius_km": float(args.same_time_merge_radius_km),
        "max_speed_kmh": float(args.max_speed_kmh),
        "max_link_km": float(args.max_link_km),
        "max_gap_steps": int(args.max_gap_steps),
        "base_time_hours": float(args.base_time_hours),
        "max_match_radius_km": float(args.max_match_radius_km),
        "min_overlap_points": int(args.min_overlap_points),
        "min_overlap_fraction": float(args.min_overlap_fraction),
        "min_continuous_overlap_hours": float(args.min_continuous_overlap_hours),
        "overlap_denominator": str(args.overlap_denominator),
        "pre_filter_counts": {
            "n_input_detections": int(len(detections)),
            "n_filtered_detections_final": int(len(filtered_detections)),
            "n_pred_tracks_raw_final": int(len(pred_summary_raw)),
            "n_truth_tracks_raw": int(truth_points_raw["sid"].nunique()) if not truth_points_raw.empty else 0,
        },
        "post_filter_counts": {
            "n_pred_tracks_filtered_final": int(len(pred_summary)),
            "n_truth_tracks_filtered": int(truth_points["sid"].nunique()) if not truth_points.empty else 0,
            "n_pred_tracks_excluded_final": int(len(pred_excluded)),
            "n_truth_tracks_excluded": int(len(truth_excluded)),
        },
        "grid_lat_name": lat_name,
        "grid_lon_name": lon_name,
        "overall_metrics": overall,
    }
    with open(out_dir / "run_summary.json", "w") as f:
        json.dump(run_summary, f, indent=2)

    print("Done.")
    print(f"Final threshold: {final_threshold:.3f}")
    print(f"Selection reason: {threshold_selection_reason}")
    print(json.dumps(overall, indent=2))
    print(f"Outputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
