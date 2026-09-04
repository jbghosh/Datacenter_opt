from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, List, Any
from collections import defaultdict
import json
import math
import time
import zipfile

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StudyConfig:
    # Grid-event discretization
    slot_minutes: int = 15
    event_hours: int = 6
    n_events: int = 20
    min_event_separation_hours: int = 24

    # Empirical Alibaba workload portfolio
    gpu_model: str = "A100-SXM4-80GB"
    job_duration_min_minutes: float = 90.0
    job_duration_max_minutes: float = 180.0
    n_jobs_small: int = 13       # 8--16 GPUs
    n_jobs_medium: int = 4       # 17--32 GPUs
    n_jobs_large: int = 1        # 33--64 GPUs
    seed_jobs: int = 883
    field_cluster_gpus: int = 256

    # Field-calibrated DVFS modes from Emerald AI / Nature Energy artefacts
    dvfs_caps_W: tuple = (250, 300, 350, 400)
    # A deliberately conservative throughput mapping is used: the minimum measured
    # normalized throughput across the eight A100 workload configurations at each cap.
    throughput_statistic: str = "minimum"
    power_statistic: str = "mean"

    # Grid-to-cluster scaling. Colangelo et al. demonstrated a 25% modulation on a
    # 256-A100 field cluster. We use the same magnitude as the peak flexible envelope.
    field_flex_fraction: float = 0.25
    target_fraction_of_envelope: float = 0.80

    # QUBO same-job conflict penalty. A sufficient bound is established analytically:
    # lambda_once > T * target_fraction^2 makes any duplicate-start solution worse
    # than the all-zero schedule because the tracking objective is nonnegative.
    lambda_once: float = 20.0

    # Initial solver tests
    cplex_time_limit_s: float = 30.0
    cplex_mip_gap: float = 1e-2
    sa_num_reads: int = 1000
    sa_num_sweeps: int = 5000
    sa_seed: int = 20260903

    @property
    def T(self) -> int:
        return int(round(self.event_hours * 60 / self.slot_minutes))

    @property
    def n_jobs(self) -> int:
        return self.n_jobs_small + self.n_jobs_medium + self.n_jobs_large

    @property
    def lambda_sufficient_bound(self) -> float:
        return self.T * self.target_fraction_of_envelope**2


DEFAULT_CONFIG = StudyConfig()


def _mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def extract_spot_trace(cluster_zip: str | Path, output_dir: str | Path) -> Tuple[Path, Path, Path]:
    """Extract only the Alibaba 2026 Spot-GPU README/job/node files."""
    cluster_zip = Path(cluster_zip)
    out = _mkdir(Path(output_dir))
    suffix_map = {
        "cluster-trace-v2026-spot-gpu/README.md": "README.md",
        "cluster-trace-v2026-spot-gpu/job_info_df.csv": "job_info_df.csv",
        "cluster-trace-v2026-spot-gpu/node_info_df.csv": "node_info_df.csv",
    }
    with zipfile.ZipFile(cluster_zip, "r") as zf:
        names = zf.namelist()
        resolved = {}
        for suffix, dst in suffix_map.items():
            matches = [n for n in names if n.endswith(suffix)]
            if not matches:
                raise FileNotFoundError(f"Alibaba ZIP is missing a file ending in {suffix}")
            resolved[matches[0]] = dst
        for src, dst in resolved.items():
            target = out / dst
            if not target.exists():
                target.write_bytes(zf.read(src))
    return out / "job_info_df.csv", out / "node_info_df.csv", out / "README.md"


def extract_emerald_data(emerald_zip: str | Path, output_dir: str | Path) -> Dict[str, Path]:
    """Extract the field-trial data needed for calibration from the Emerald AI archive."""
    emerald_zip = Path(emerald_zip)
    out = _mkdir(Path(output_dir))
    wanted = {
        "data/dvfs_sweep.csv": "dvfs_sweep.csv",
        "data/SRP_total_power.csv": "SRP_total_power.csv",
        "README.md": "README_EMERALD.md",
    }
    result = {}
    with zipfile.ZipFile(emerald_zip, "r") as zf:
        names = zf.namelist()
        for suffix, dst in wanted.items():
            matches = [n for n in names if n.endswith(suffix)]
            if not matches:
                raise FileNotFoundError(f"Emerald ZIP is missing a file ending in {suffix}")
            target = out / dst
            target.write_bytes(zf.read(matches[0]))
            result[dst] = target
    return result


def build_dvfs_calibration(dvfs_csv: str | Path,
                           cfg: StudyConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Build a field-calibrated A100 power/throughput table.

    The Emerald artefact contains eight A100 workload configurations measured at power
    caps from 100 to 400 W. For unknown Alibaba Spot workloads we use:
      * mean measured power per GPU at each selected cap;
      * minimum normalized throughput across the eight measured workloads at each cap.

    The throughput choice is conservative: if an admitted job is assigned a cap, its
    event-window runtime is inflated by 1/min_throughput before discretization.
    """
    raw = pd.read_csv(dvfs_csv)
    rows = []
    for cap in cfg.dvfs_caps_W:
        g = raw[pd.to_numeric(raw["GPU power cap"], errors="coerce") == cap].copy()
        if len(g) == 0:
            raise RuntimeError(f"DVFS sweep has no observations at {cap} W")
        rows.append({
            "power_cap_W": int(cap),
            "n_workload_profiles": int(len(g)),
            "power_per_gpu_W_mean": float(g["power per GPU"].mean()),
            "power_per_gpu_W_median": float(g["power per GPU"].median()),
            "power_per_gpu_W_min": float(g["power per GPU"].min()),
            "power_per_gpu_W_max": float(g["power per GPU"].max()),
            "throughput_min": float(g["normalized throughput"].min()),
            "throughput_mean": float(g["normalized throughput"].mean()),
            "throughput_median": float(g["normalized throughput"].median()),
            "throughput_max": float(g["normalized throughput"].max()),
        })
    cal = pd.DataFrame(rows).sort_values("power_cap_W").reset_index(drop=True)
    cal["model_power_per_gpu_kW"] = cal["power_per_gpu_W_mean"] / 1000.0
    cal["model_throughput_factor"] = cal["throughput_min"]
    return cal


def field_cluster_power_summary(srp_total_power_csv: str | Path,
                                dvfs_calibration: pd.DataFrame,
                                cfg: StudyConfig = DEFAULT_CONFIG) -> Dict[str, float]:
    """Return two independent field-calibrated full-cluster power estimates."""
    srp = pd.read_csv(srp_total_power_csv)
    srp["timestamp"] = pd.to_datetime(srp["timestamp"], errors="coerce")
    # Use the pre-event interval before the SRP field-trial ramp begins.
    hhmm = srp["timestamp"].dt.strftime("%H:%M")
    pre = srp[(hhmm >= "14:40") & (hhmm <= "16:00")]
    pre_mean_kw = float(pre["total"].mean() / 1000.0)
    pre_median_kw = float(pre["total"].median() / 1000.0)
    p400 = float(dvfs_calibration.loc[dvfs_calibration.power_cap_W == 400,
                                      "model_power_per_gpu_kW"].iloc[0])
    dvfs_implied_kw = cfg.field_cluster_gpus * p400
    return {
        "srp_pre_event_mean_kW": pre_mean_kw,
        "srp_pre_event_median_kW": pre_median_kw,
        "dvfs_implied_256gpu_full_power_kW": float(dvfs_implied_kw),
        "field_flexible_envelope_peak_kW": float(cfg.field_flex_fraction * dvfs_implied_kw),
    }


def reconstruct_caiso_timestamp(df: pd.DataFrame) -> pd.Series:
    date0 = pd.to_datetime(df["Date"]).dt.normalize()
    return (
        date0
        + pd.to_timedelta(df["Hour"].astype(int) - 1, unit="h")
        + pd.to_timedelta((df["Interval"].astype(int) - 1) * 5, unit="m")
    )


def load_caiso_workbook(xlsx_path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cur = pd.read_excel(xlsx_path, sheet_name="Curtailments")
    prod = pd.read_excel(xlsx_path, sheet_name="Production")
    cur["timestamp"] = reconstruct_caiso_timestamp(cur)
    prod["timestamp"] = reconstruct_caiso_timestamp(prod)
    cur["wind_curtailment_MW"] = pd.to_numeric(cur["Wind Curtailment"], errors="coerce").fillna(0.0)
    cur["solar_curtailment_MW"] = pd.to_numeric(cur["Solar Curtailment"], errors="coerce").fillna(0.0)
    cur["curtailment_MW"] = cur["wind_curtailment_MW"] + cur["solar_curtailment_MW"]
    cur5 = cur.groupby("timestamp", as_index=False).agg(
        wind_curtailment_MW=("wind_curtailment_MW", "sum"),
        solar_curtailment_MW=("solar_curtailment_MW", "sum"),
        curtailment_MW=("curtailment_MW", "sum"),
        n_reason_records=("Reason", "size"),
    )
    prod_cols = {
        "Load": "load_MW", "Solar": "solar_MW", "Wind": "wind_MW",
        "Net Load": "net_load_MW", "Renewables": "renewables_MW",
    }
    keep = ["timestamp"]
    for old, new in prod_cols.items():
        prod[new] = pd.to_numeric(prod[old], errors="coerce")
        keep.append(new)
    return cur5, prod[keep].copy()


def prepare_caiso_15min(xlsx_path: str | Path, cfg: StudyConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    cur5, prod5 = load_caiso_workbook(xlsx_path)
    start = min(cur5.timestamp.min(), prod5.timestamp.min()).floor("D")
    end = max(cur5.timestamp.max(), prod5.timestamp.max()).ceil("D") - pd.Timedelta(minutes=5)
    idx = pd.date_range(start, end, freq="5min")
    base = pd.DataFrame({"timestamp": idx}).merge(cur5, on="timestamp", how="left").merge(prod5, on="timestamp", how="left")
    for c in ["wind_curtailment_MW", "solar_curtailment_MW", "curtailment_MW", "n_reason_records"]:
        base[c] = base[c].fillna(0.0)
    out = base.set_index("timestamp").resample(f"{cfg.slot_minutes}min").mean(numeric_only=True)
    out["curtailment_energy_MWh"] = out["curtailment_MW"] * cfg.slot_minutes / 60.0
    return out.reset_index()


def select_caiso_events(caiso15: pd.DataFrame, cfg: StudyConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    df = caiso15.set_index("timestamp").copy()
    T = cfg.T
    rolling_energy = df["curtailment_energy_MWh"].rolling(T, min_periods=T).sum()
    candidates = []
    for end_ts, score in rolling_energy.dropna().sort_values(ascending=False).items():
        start_ts = end_ts - pd.Timedelta(minutes=cfg.slot_minutes * (T - 1))
        candidates.append((start_ts, end_ts, float(score)))
    selected = []
    sep = pd.Timedelta(hours=cfg.min_event_separation_hours)
    for st, en, score in candidates:
        if all(abs(st - st2) >= sep for st2, _, _ in selected):
            selected.append((st, en, score))
        if len(selected) >= cfg.n_events:
            break
    if len(selected) < cfg.n_events:
        raise RuntimeError(f"Found only {len(selected)} separated events, need {cfg.n_events}.")
    rows = []
    for event_id, (st, en, score) in enumerate(selected):
        sub = df.loc[st:en]
        rows.append({
            "event_id": event_id, "start": st, "end": en,
            "raw_curtailment_MWh": score,
            "mean_curtailment_MW": float(sub.curtailment_MW.mean()),
            "max_curtailment_MW": float(sub.curtailment_MW.max()),
            "min_curtailment_MW": float(sub.curtailment_MW.min()),
            "fraction_positive": float((sub.curtailment_MW > 0).mean()),
        })
    return pd.DataFrame(rows)


def build_event_profiles(caiso15: pd.DataFrame, events: pd.DataFrame,
                         flexible_envelope_peak_kW: float,
                         cfg: StudyConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    df = caiso15.set_index("timestamp")
    selected_max = max(float(df.loc[r.start:r.end, "curtailment_MW"].max()) for r in events.itertuples())
    scale_kw_per_MW = flexible_envelope_peak_kW / selected_max if selected_max > 0 else 0.0
    rows = []
    for r in events.itertuples():
        sub = df.loc[r.start:r.end].copy()
        if len(sub) != cfg.T:
            raise RuntimeError(f"Event {r.event_id} has {len(sub)} slots, expected {cfg.T}")
        env = sub["curtailment_MW"].to_numpy(float) * scale_kw_per_MW
        target = cfg.target_fraction_of_envelope * env
        for slot, (ts, raw_mw, a_kw, t_kw) in enumerate(zip(sub.index, sub.curtailment_MW, env, target)):
            rows.append({
                "event_id": int(r.event_id), "slot": slot, "timestamp": ts,
                "raw_curtailment_MW": float(raw_mw),
                "allocated_curtailment_kW": float(a_kw),
                "dispatch_target_kW": float(t_kw),
                "global_scale_kW_per_MW": float(scale_kw_per_MW),
                "flexible_envelope_peak_kW": float(flexible_envelope_peak_kW),
            })
    return pd.DataFrame(rows)


def load_alibaba_spot_jobs(job_csv: str | Path, cfg: StudyConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    jobs = pd.read_csv(job_csv)
    jobs = jobs[(jobs["job_type"] == "Spot") & (jobs["gpu_model"] == cfg.gpu_model)].copy()
    jobs["total_gpus"] = pd.to_numeric(jobs["gpu_request"], errors="coerce") * pd.to_numeric(jobs["worker_num"], errors="coerce")
    jobs["duration_minutes"] = pd.to_numeric(jobs["duration"], errors="coerce") / 60.0
    jobs = jobs[
        jobs["total_gpus"].between(8, 64)
        & jobs["duration_minutes"].between(cfg.job_duration_min_minutes, cfg.job_duration_max_minutes)
        & np.isclose(jobs["total_gpus"], np.round(jobs["total_gpus"]))
    ].copy()
    jobs["total_gpus"] = jobs["total_gpus"].round().astype(int)
    return jobs


def select_fixed_job_portfolio(filtered_jobs: pd.DataFrame,
                               cfg: StudyConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    # Separate seeds make the selection deterministic and auditable.
    specs = [
        (8, 16, cfg.n_jobs_small, cfg.seed_jobs),
        (17, 32, cfg.n_jobs_medium, cfg.seed_jobs + 1000),
        (33, 64, cfg.n_jobs_large, cfg.seed_jobs + 2000),
    ]
    parts = []
    for lo, hi, n, seed in specs:
        sub = filtered_jobs[filtered_jobs.total_gpus.between(lo, hi)]
        if len(sub) < n:
            raise RuntimeError(f"Not enough jobs in GPU bin {lo}-{hi}: have {len(sub)}, need {n}")
        parts.append(sub.sample(n=n, random_state=seed, replace=False))
    out = pd.concat(parts, ignore_index=True)
    out.insert(0, "portfolio_job_id", np.arange(len(out), dtype=int))
    total = int(out.total_gpus.sum())
    if total != cfg.field_cluster_gpus:
        raise RuntimeError(
            f"Deterministic portfolio uses {total} GPUs, expected exactly {cfg.field_cluster_gpus}. "
            "Do not silently change the trace or seed before the D-Wave campaign."
        )
    keep = [
        "portfolio_job_id", "job_name", "organization", "gpu_model", "gpu_request",
        "worker_num", "total_gpus", "cpu_request", "submit_time", "duration",
        "duration_minutes", "job_type",
    ]
    return out[keep].sort_values("portfolio_job_id").reset_index(drop=True)


def build_candidate_table(portfolio: pd.DataFrame, dvfs_calibration: pd.DataFrame,
                          cfg: StudyConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    cal = dvfs_calibration.set_index("power_cap_W")
    rows = []
    v = 0
    for r in portfolio.itertuples():
        for cap in cfg.dvfs_caps_W:
            p_gpu = float(cal.loc[cap, "model_power_per_gpu_kW"])
            thr = float(cal.loc[cap, "model_throughput_factor"])
            adjusted_minutes = float(r.duration_minutes) / thr
            L = int(math.ceil(adjusted_minutes / cfg.slot_minutes))
            if L > cfg.T:
                continue
            for s in range(cfg.T - L + 1):
                rows.append({
                    "var_index": v,
                    "var_name": f"x_j{int(r.portfolio_job_id):02d}_p{int(cap)}_s{s:02d}",
                    "portfolio_job_id": int(r.portfolio_job_id),
                    "power_cap_W": int(cap),
                    "start_slot": int(s),
                    "end_slot_exclusive": int(s + L),
                    "duration_slots": int(L),
                    "baseline_duration_minutes": float(r.duration_minutes),
                    "mode_adjusted_duration_minutes": adjusted_minutes,
                    "model_throughput_factor": thr,
                    "total_gpus": int(r.total_gpus),
                    "model_power_per_gpu_kW": p_gpu,
                    "job_power_kW": float(r.total_gpus) * p_gpu,
                })
                v += 1
    return pd.DataFrame(rows)


def _add_linear(d: Dict[int, float], i: int, value: float) -> None:
    d[i] = d.get(i, 0.0) + float(value)


def _add_quad(d: Dict[Tuple[int, int], float], i: int, j: int, value: float) -> None:
    if i == j:
        raise ValueError("Off-diagonal quadratic dictionary received i==j")
    if i > j:
        i, j = j, i
    d[(i, j)] = d.get((i, j), 0.0) + float(value)


def build_qubo(portfolio: pd.DataFrame, dvfs_calibration: pd.DataFrame,
               event_profile: pd.DataFrame, cfg: StudyConfig = DEFAULT_CONFIG) -> Dict[str, Any]:
    """Build the field-calibrated start-time + DVFS-mode QUBO.

    Candidate x_{j,m,s}=1 means empirical Spot job j is admitted, starts at s and runs
    at one fixed A100 power cap m until its baseline work completes. Mode-dependent
    completion time is inflated using the conservative field-measured throughput factor.

    H = sum_t [(P_GPU(t)-P_target(t))/P_flex]^2
        + lambda_once * sum_j sum_{a<b in candidates(j)} x_a x_b.

    The same-job penalty has a rigorous sufficient bound: because the zero schedule has
    H_track <= T*eta^2 and H_track >= 0, lambda_once > T*eta^2 ensures a global optimum
    cannot contain even one duplicate pair.
    """
    ep = event_profile.sort_values("slot")
    if len(ep) != cfg.T:
        raise ValueError(f"Expected {cfg.T} event slots, got {len(ep)}")
    if not cfg.lambda_once > cfg.lambda_sufficient_bound:
        raise ValueError(
            f"lambda_once={cfg.lambda_once} must exceed sufficient bound "
            f"T*eta^2={cfg.lambda_sufficient_bound:.6g}"
        )
    vt = build_candidate_table(portfolio, dvfs_calibration, cfg)
    n = len(vt)
    linear: Dict[int, float] = {}
    quadratic: Dict[Tuple[int, int], float] = {}
    target = ep.dispatch_target_kW.to_numpy(float)
    P0 = float(ep.flexible_envelope_peak_kW.iloc[0])
    offset = float(np.sum((target / P0) ** 2))

    active_by_t: List[List[Tuple[int, float]]] = [[] for _ in range(cfg.T)]
    for r in vt.itertuples():
        a = float(r.job_power_kW) / P0
        for t in range(int(r.start_slot), int(r.end_slot_exclusive)):
            active_by_t[t].append((int(r.var_index), a))

    for t in range(cfg.T):
        rt = float(target[t] / P0)
        active = active_by_t[t]
        for v, a in active:
            _add_linear(linear, v, a*a - 2.0*rt*a)
        for ii in range(len(active)):
            vi, ai = active[ii]
            for jj in range(ii + 1, len(active)):
                vj, aj = active[jj]
                _add_quad(quadratic, vi, vj, 2.0*ai*aj)

    for _, g in vt.groupby("portfolio_job_id"):
        vs = g.var_index.astype(int).tolist()
        for ii in range(len(vs)):
            for jj in range(ii + 1, len(vs)):
                _add_quad(quadratic, vs[ii], vs[jj], cfg.lambda_once)

    linear = {i: v for i, v in linear.items() if abs(v) > 1e-15}
    quadratic = {ij: v for ij, v in quadratic.items() if abs(v) > 1e-15}
    return {
        "linear": linear, "quadratic": quadratic, "offset": offset,
        "var_table": vt, "n_variables": n, "n_edges": len(quadratic),
        "density": len(quadratic)/(n*(n-1)/2) if n > 1 else 0.0,
        "event_id": int(ep.event_id.iloc[0]), "power_scale_kW": P0,
        "zero_schedule_objective": offset,
        "lambda_sufficient_bound": cfg.lambda_sufficient_bound,
    }


def qubo_energy(sample: np.ndarray, qubo: Dict[str, Any], include_offset: bool = True) -> float:
    x = np.asarray(sample, dtype=float)
    e = float(qubo["offset"] if include_offset else 0.0)
    for i, b in qubo["linear"].items():
        e += b*x[i]
    for (i, j), q in qubo["quadratic"].items():
        e += q*x[i]*x[j]
    return float(e)


def decode_sample(sample: np.ndarray, portfolio: pd.DataFrame,
                  event_profile: pd.DataFrame, qubo: Dict[str, Any],
                  cfg: StudyConfig = DEFAULT_CONFIG) -> Dict[str, Any]:
    x = np.asarray(sample, dtype=int)
    vt = qubo["var_table"]
    chosen = vt.loc[x[vt.var_index.to_numpy(dtype=int)] > 0].copy()
    load = np.zeros(cfg.T, dtype=float)
    gpu_count = np.zeros(cfg.T, dtype=float)
    for r in chosen.itertuples():
        sl = slice(int(r.start_slot), int(r.end_slot_exclusive))
        load[sl] += float(r.job_power_kW)
        gpu_count[sl] += int(r.total_gpus)

    counts = chosen.groupby("portfolio_job_id").size() if len(chosen) else pd.Series(dtype=int)
    duplicate_starts = int(np.sum(np.maximum(counts.to_numpy() - 1, 0))) if len(counts) else 0
    ep = event_profile.sort_values("slot")
    target = ep.dispatch_target_kW.to_numpy(float)
    envelope = ep.allocated_curtailment_kW.to_numpy(float)
    P0 = float(ep.flexible_envelope_peak_kW.iloc[0])
    tracking = float(np.sum(((load-target)/P0)**2))
    penalty = float(cfg.lambda_once * sum(math.comb(int(c), 2) for c in counts if c >= 2))
    direct = tracking + penalty
    dt_h = cfg.slot_minutes/60.0
    absorbed = np.minimum(load, envelope)

    selected_jobs = portfolio.set_index("portfolio_job_id").loc[counts.index] if len(counts) else portfolio.iloc[0:0]
    useful_baseline_gpu_hours = float(
        np.sum(selected_jobs.total_gpus.to_numpy(float) * selected_jobs.duration_minutes.to_numpy(float)/60.0)
    ) if len(selected_jobs) else 0.0

    cap_counts = chosen.groupby("power_cap_W").size().to_dict() if len(chosen) else {}
    metrics = {
        "objective_direct": direct,
        "tracking_objective": tracking,
        "once_penalty": penalty,
        "duplicate_starts": duplicate_starts,
        "n_jobs_scheduled": int((counts > 0).sum()) if len(counts) else 0,
        "tracking_RMSE_kW": float(np.sqrt(np.mean((load-target)**2))),
        "max_envelope_violation_kW": float(np.maximum(load-envelope, 0.0).max()),
        "envelope_violation_kWh": float(np.maximum(load-envelope, 0.0).sum()*dt_h),
        "max_concurrent_gpus": int(gpu_count.max()),
        "cluster_gpu_capacity": int(cfg.field_cluster_gpus),
        "allocated_curtailment_kWh": float(envelope.sum()*dt_h),
        "target_energy_kWh": float(target.sum()*dt_h),
        "scheduled_gpu_energy_kWh": float(load.sum()*dt_h),
        "renewable_absorbed_kWh": float(absorbed.sum()*dt_h),
        "capture_fraction_of_envelope": float(absorbed.sum()/envelope.sum()) if envelope.sum() > 0 else 0.0,
        "target_energy_ratio": float(load.sum()/target.sum()) if target.sum() > 0 else 0.0,
        "completed_baseline_gpu_hours": useful_baseline_gpu_hours,
        "cap250_jobs": int(cap_counts.get(250, 0)),
        "cap300_jobs": int(cap_counts.get(300, 0)),
        "cap350_jobs": int(cap_counts.get(350, 0)),
        "cap400_jobs": int(cap_counts.get(400, 0)),
    }
    return {"chosen": chosen, "load_kW": load, "gpu_count": gpu_count, "metrics": metrics}


def random_equivalence_tests(qubo: Dict[str, Any], portfolio: pd.DataFrame,
                             event_profile: pd.DataFrame, cfg: StudyConfig = DEFAULT_CONFIG,
                             n_tests: int = 20, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    n = int(qubo["n_variables"])
    for k in range(n_tests):
        x = (rng.random(n) < 0.025).astype(np.int8)
        dec = decode_sample(x, portfolio, event_profile, qubo, cfg)
        eq = qubo_energy(x, qubo, include_offset=True)
        ed = dec["metrics"]["objective_direct"]
        rows.append({"test": k, "direct": ed, "qubo": eq, "abs_error": abs(ed-eq)})
    return pd.DataFrame(rows)


def build_dimod_bqm(qubo: Dict[str, Any]):
    try:
        import dimod
    except Exception as e:
        raise ImportError("dimod is required. Install with: pip install dimod dwave-neal") from e
    return dimod.BinaryQuadraticModel(qubo["linear"], qubo["quadratic"], qubo["offset"], dimod.BINARY)


def solve_sa_qubo(qubo: Dict[str, Any], cfg: StudyConfig = DEFAULT_CONFIG,
                  num_reads: int | None = None, num_sweeps: int | None = None,
                  seed: int | None = None) -> Dict[str, Any]:
    try:
        import neal
    except Exception as e:
        raise ImportError("dwave-neal is required. Install with: pip install dwave-neal dimod") from e
    bqm = build_dimod_bqm(qubo)
    sampler = neal.SimulatedAnnealingSampler()
    t0 = time.time()
    ss = sampler.sample(
        bqm,
        num_reads=num_reads or cfg.sa_num_reads,
        num_sweeps=num_sweeps or cfg.sa_num_sweeps,
        seed=cfg.sa_seed if seed is None else seed,
    )
    wall = time.time() - t0
    best = ss.first
    sample = np.array([int(best.sample[i]) for i in range(qubo["n_variables"])], dtype=np.int8)
    return {"sample": sample, "energy": float(best.energy), "wall_time_s": wall, "status": "ok", "sampleset": ss}


def solve_cplex_qubo(qubo: Dict[str, Any], cfg: StudyConfig = DEFAULT_CONFIG,
                     time_limit_s: float | None = None, log_output: bool = False) -> Dict[str, Any]:
    try:
        from docplex.mp.model import Model
    except Exception as e:
        raise ImportError("docplex + IBM CPLEX Python API are required.") from e
    mdl = Model(name=f"spot_gpu_dvfs_qubo_event_{qubo['event_id']}")
    x = mdl.binary_var_list(qubo["n_variables"], name="x")
    expr_lin = mdl.sum(coef*x[i] for i, coef in qubo["linear"].items())
    expr_quad = mdl.sum(coef*x[i]*x[j] for (i, j), coef in qubo["quadratic"].items())
    mdl.minimize(expr_lin + expr_quad + qubo["offset"])
    mdl.parameters.timelimit = float(time_limit_s or cfg.cplex_time_limit_s)
    mdl.parameters.mip.tolerances.mipgap = float(cfg.cplex_mip_gap)
    t0 = time.time(); sol = mdl.solve(log_output=log_output); wall = time.time()-t0
    details = mdl.solve_details
    if sol is None:
        return {"sample": None, "energy": np.nan, "wall_time_s": wall,
                "status": str(details.status), "best_bound": np.nan, "mip_gap": np.nan}
    sample = np.array([int(round(sol.get_value(v))) for v in x], dtype=np.int8)
    return {
        "sample": sample, "energy": float(sol.objective_value), "wall_time_s": wall,
        "status": str(details.status),
        "best_bound": float(getattr(details, "best_bound", np.nan)),
        "mip_gap": float(getattr(details, "mip_relative_gap", np.nan)),
    }


def solve_cplex_constrained_reference(portfolio: pd.DataFrame, dvfs_calibration: pd.DataFrame,
                                      event_profile: pd.DataFrame,
                                      cfg: StudyConfig = DEFAULT_CONFIG,
                                      time_limit_s: float | None = None,
                                      log_output: bool = False) -> Dict[str, Any]:
    """Native constrained MIQP using the same 990 candidate binaries and squared tracking objective."""
    try:
        from docplex.mp.model import Model
    except Exception as e:
        raise ImportError("docplex + IBM CPLEX Python API are required.") from e
    vt = build_candidate_table(portfolio, dvfs_calibration, cfg)
    ep = event_profile.sort_values("slot")
    mdl = Model(name=f"spot_gpu_dvfs_constrained_event_{int(ep.event_id.iloc[0])}")
    x = mdl.binary_var_list(len(vt), name="x")

    # Each empirical Spot job can be admitted at most once, with one start and one DVFS mode.
    for _, g in vt.groupby("portfolio_job_id"):
        inds = g.var_index.astype(int).tolist()
        mdl.add_constraint(mdl.sum(x[i] for i in inds) <= 1)

    target = ep.dispatch_target_kW.to_numpy(float)
    envelope = ep.allocated_curtailment_kW.to_numpy(float)
    P0 = float(ep.flexible_envelope_peak_kW.iloc[0])
    loads = []
    gpu_loads = []
    for t in range(cfg.T):
        act = vt[(vt.start_slot <= t) & (vt.end_slot_exclusive > t)]
        pexpr = mdl.sum(float(r.job_power_kW)*x[int(r.var_index)] for r in act.itertuples())
        gexpr = mdl.sum(float(r.total_gpus)*x[int(r.var_index)] for r in act.itertuples())
        loads.append(pexpr); gpu_loads.append(gexpr)
        mdl.add_constraint(pexpr <= float(envelope[t]))
        # Redundant for the frozen 256-GPU portfolio, but explicit for auditability.
        mdl.add_constraint(gexpr <= cfg.field_cluster_gpus)

    mdl.minimize(mdl.sum(((loads[t]-float(target[t]))/P0)**2 for t in range(cfg.T)))
    mdl.parameters.timelimit = float(time_limit_s or cfg.cplex_time_limit_s)
    mdl.parameters.mip.tolerances.mipgap = float(cfg.cplex_mip_gap)
    t0 = time.time(); sol = mdl.solve(log_output=log_output); wall = time.time()-t0
    details = mdl.solve_details
    if sol is None:
        return {"sample": None, "energy": np.nan, "wall_time_s": wall,
                "status": str(details.status), "best_bound": np.nan, "mip_gap": np.nan}
    sample = np.array([int(round(sol.get_value(v))) for v in x], dtype=np.int8)
    return {
        "sample": sample, "energy": float(sol.objective_value), "wall_time_s": wall,
        "status": str(details.status),
        "best_bound": float(getattr(details, "best_bound", np.nan)),
        "mip_gap": float(getattr(details, "mip_relative_gap", np.nan)),
    }


def greedy_physical_smoketest(portfolio: pd.DataFrame, dvfs_calibration: pd.DataFrame,
                              event_profile: pd.DataFrame,
                              cfg: StudyConfig = DEFAULT_CONFIG) -> Dict[str, Any]:
    """Deterministic physical feasibility smoke test; not a publication solver."""
    vt = build_candidate_table(portfolio, dvfs_calibration, cfg)
    ep = event_profile.sort_values("slot")
    target = ep.dispatch_target_kW.to_numpy(float)
    envelope = ep.allocated_curtailment_kW.to_numpy(float)
    P0 = float(ep.flexible_envelope_peak_kW.iloc[0])
    load = np.zeros(cfg.T)
    used = set(); selected = []
    arrays = []
    for r in vt.itertuples():
        a = np.zeros(cfg.T)
        a[int(r.start_slot):int(r.end_slot_exclusive)] = float(r.job_power_kW)
        arrays.append(a)
    obj = lambda z: float(np.sum(((z-target)/P0)**2))
    current = obj(load)
    while True:
        best = None
        for idx, r in enumerate(vt.itertuples()):
            if int(r.portfolio_job_id) in used:
                continue
            trial = load + arrays[idx]
            if np.any(trial > envelope + 1e-12):
                continue
            val = obj(trial)
            if best is None or val < best[0]:
                best = (val, idx)
        if best is None or best[0] >= current - 1e-12:
            break
        current, idx = best
        r = vt.iloc[idx]
        load += arrays[idx]
        used.add(int(r.portfolio_job_id))
        selected.append(idx)
    x = np.zeros(len(vt), dtype=np.int8); x[selected] = 1
    q = build_qubo(portfolio, dvfs_calibration, event_profile, cfg)
    return {"sample": x, "decoded": decode_sample(x, portfolio, event_profile, q, cfg)}


def prepare_from_original_sources(caiso_xlsx: str | Path, cluster_zip: str | Path,
                                  emerald_zip: str | Path, output_dir: str | Path,
                                  cfg: StudyConfig = DEFAULT_CONFIG) -> Dict[str, Path]:
    out = _mkdir(Path(output_dir))
    ali = _mkdir(out/"alibaba_extract")
    emd = _mkdir(out/"emerald_extract")
    job_csv, node_csv, ali_readme = extract_spot_trace(cluster_zip, ali)
    emerald = extract_emerald_data(emerald_zip, emd)
    caiso15 = prepare_caiso_15min(caiso_xlsx, cfg)
    return prepare_from_tables(caiso15, job_csv, emerald["dvfs_sweep.csv"],
                               emerald["SRP_total_power.csv"], out, cfg,
                               source_metadata={
                                   "input_caiso": str(Path(caiso_xlsx).resolve()),
                                   "input_alibaba_zip": str(Path(cluster_zip).resolve()),
                                   "input_emerald_zip": str(Path(emerald_zip).resolve()),
                               })


def prepare_from_tables(caiso15: pd.DataFrame, alibaba_job_csv: str | Path,
                        dvfs_csv: str | Path, srp_power_csv: str | Path,
                        output_dir: str | Path, cfg: StudyConfig = DEFAULT_CONFIG,
                        source_metadata: Dict[str, Any] | None = None) -> Dict[str, Path]:
    out = _mkdir(Path(output_dir))
    cal = build_dvfs_calibration(dvfs_csv, cfg)
    field = field_cluster_power_summary(srp_power_csv, cal, cfg)
    events = select_caiso_events(caiso15, cfg)
    profiles = build_event_profiles(caiso15, events, field["field_flexible_envelope_peak_kW"], cfg)
    filtered = load_alibaba_spot_jobs(alibaba_job_csv, cfg)
    portfolio = select_fixed_job_portfolio(filtered, cfg)
    vt = build_candidate_table(portfolio, cal, cfg)

    cal_path = out/"emerald_dvfs_calibration.csv"; cal.to_csv(cal_path, index=False)
    caiso_path = out/"caiso_15min.csv"; caiso15.to_csv(caiso_path, index=False)
    events_path = out/"selected_caiso_events.csv"; events.to_csv(events_path, index=False)
    profiles_path = out/"event_profiles_long.csv"; profiles.to_csv(profiles_path, index=False)
    portfolio_path = out/"benchmark_spot_jobs_dvfs.csv"; portfolio.to_csv(portfolio_path, index=False)
    vt_path = out/"candidate_variable_table.csv"; vt.to_csv(vt_path, index=False)

    q0 = build_qubo(portfolio, cal, profiles[profiles.event_id == 0], cfg)
    eq = random_equivalence_tests(q0, portfolio, profiles[profiles.event_id == 0], cfg, n_tests=12, seed=991)
    eq_path = out/"qubo_equivalence_smoketest.csv"; eq.to_csv(eq_path, index=False)

    greedy_rows = []
    for eid in [0, 1, 2]:
        ep = profiles[profiles.event_id == eid]
        gr = greedy_physical_smoketest(portfolio, cal, ep, cfg)
        row = {"event_id": eid, **gr["decoded"]["metrics"]}
        greedy_rows.append(row)
    greedy_path = out/"greedy_physical_smoketest.csv"; pd.DataFrame(greedy_rows).to_csv(greedy_path, index=False)

    p400 = float(cal.loc[cal.power_cap_W == 400, "model_power_per_gpu_kW"].iloc[0])
    full400_kWh = 0.0
    for r in portfolio.itertuples():
        L = math.ceil(float(r.duration_minutes)/cfg.slot_minutes)
        full400_kWh += int(r.total_gpus)*p400*L*(cfg.slot_minutes/60.0)

    manifest = {
        "config": asdict(cfg),
        **(source_metadata or {}),
        "selected_jobs": int(len(portfolio)),
        "selected_portfolio_total_gpus": int(portfolio.total_gpus.sum()),
        "eligible_alibaba_jobs": int(len(filtered)),
        "dvfs_modes_W": list(cfg.dvfs_caps_W),
        "field_cluster_power": field,
        "full_portfolio_400W_mode_energy_kWh": float(full400_kWh),
        "qubo_variables": int(q0["n_variables"]),
        "qubo_edges": int(q0["n_edges"]),
        "qubo_density": float(q0["density"]),
        "zero_schedule_objective_event0": float(q0["zero_schedule_objective"]),
        "lambda_sufficient_bound": float(q0["lambda_sufficient_bound"]),
        "lambda_once": float(cfg.lambda_once),
        "equivalence_max_abs_error": float(eq.abs_error.max()),
        "scientific_upgrade": (
            "256-GPU field-calibrated A100 portfolio; candidate-specific DVFS power and conservative "
            "completion-time inflation from Emerald AI measurements; 25% peak flexible-power envelope "
            "anchored to the Nature Energy field demonstration; exact 256-GPU aggregate portfolio."
        ),
    }
    manifest_path = out/"revised_data_and_qubo_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    return {
        "dvfs_calibration": cal_path, "caiso15": caiso_path, "events": events_path,
        "profiles": profiles_path, "portfolio": portfolio_path, "vartable": vt_path,
        "equivalence": eq_path, "greedy": greedy_path, "manifest": manifest_path,
    }
