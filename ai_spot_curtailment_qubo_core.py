from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, List, Any, Iterable
import json
import math
import time
import zipfile

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StudyConfig:
    # Time discretization and event selection
    slot_minutes: int = 15
    event_hours: int = 6
    n_events: int = 20
    min_event_separation_hours: int = 24

    # Empirical workload portfolio
    gpu_model: str = "A100-SXM4-80GB"
    gpu_tdp_kw: float = 0.400  # NVIDIA A100 80GB SXM standard TDP = 400 W
    n_jobs_small: int = 26     # 8--16 GPUs
    n_jobs_medium: int = 16    # 17--64 GPUs
    n_jobs_large: int = 6      # 65--256 GPUs
    job_duration_min_minutes: float = 15.0
    job_duration_max_minutes: float = 180.0
    seed_jobs: int = 99

    # Grid-to-data-centre scaling
    pod_cap_kw: float = 300.0
    target_fraction_of_envelope: float = 0.80

    # QUBO
    lambda_once: float = 30.0

    # Initial solver tests
    cplex_time_limit_s: float = 30.0
    cplex_mip_gap: float = 1e-2
    sa_num_reads: int = 500
    sa_num_sweeps: int = 5000
    sa_seed: int = 20260902

    @property
    def T(self) -> int:
        return int(round(self.event_hours * 60 / self.slot_minutes))

    @property
    def n_jobs(self) -> int:
        return self.n_jobs_small + self.n_jobs_medium + self.n_jobs_large


DEFAULT_CONFIG = StudyConfig()


def _mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def extract_spot_trace(cluster_zip: str | Path, output_dir: str | Path) -> Tuple[Path, Path, Path]:
    """Extract only the 2026 spot-GPU README/job/node files from the Alibaba archive."""
    cluster_zip = Path(cluster_zip)
    out = _mkdir(Path(output_dir))
    wanted = {
        "clusterdata-master/cluster-trace-v2026-spot-gpu/README.md": "README.md",
        "clusterdata-master/cluster-trace-v2026-spot-gpu/job_info_df.csv": "job_info_df.csv",
        "clusterdata-master/cluster-trace-v2026-spot-gpu/node_info_df.csv": "node_info_df.csv",
    }
    with zipfile.ZipFile(cluster_zip, "r") as zf:
        names = set(zf.namelist())
        missing = [x for x in wanted if x not in names]
        if missing:
            raise FileNotFoundError(f"Alibaba ZIP is missing expected files: {missing}")
        for src, dst in wanted.items():
            target = out / dst
            if not target.exists():
                target.write_bytes(zf.read(src))
    return out / "job_info_df.csv", out / "node_info_df.csv", out / "README.md"


def reconstruct_caiso_timestamp(df: pd.DataFrame) -> pd.Series:
    """Rebuild five-minute timestamps from operating date + HE + interval.

    The workbook Date cells contain timezone/DST display offsets in some rows. We use only
    the calendar date from Date and reconstruct the interval from Hour (1..24) and
    Interval (1..12), which is the operational indexing supplied by CAISO.
    """
    date0 = pd.to_datetime(df["Date"]).dt.normalize()
    return (
        date0
        + pd.to_timedelta(df["Hour"].astype(int) - 1, unit="h")
        + pd.to_timedelta((df["Interval"].astype(int) - 1) * 5, unit="m")
    )


def load_caiso_workbook(xlsx_path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    xlsx_path = Path(xlsx_path)
    cur = pd.read_excel(xlsx_path, sheet_name="Curtailments")
    prod = pd.read_excel(xlsx_path, sheet_name="Production")

    cur["timestamp"] = reconstruct_caiso_timestamp(cur)
    prod["timestamp"] = reconstruct_caiso_timestamp(prod)

    cur["wind_curtailment_MW"] = pd.to_numeric(cur["Wind Curtailment"], errors="coerce").fillna(0.0)
    cur["solar_curtailment_MW"] = pd.to_numeric(cur["Solar Curtailment"], errors="coerce").fillna(0.0)
    cur["curtailment_MW"] = cur["wind_curtailment_MW"] + cur["solar_curtailment_MW"]

    # Same interval can have separate Local/System rows; these are separate curtailment records.
    cur5 = cur.groupby("timestamp", as_index=False).agg(
        wind_curtailment_MW=("wind_curtailment_MW", "sum"),
        solar_curtailment_MW=("solar_curtailment_MW", "sum"),
        curtailment_MW=("curtailment_MW", "sum"),
        n_reason_records=("Reason", "size"),
    )

    prod_cols = {
        "Load": "load_MW",
        "Solar": "solar_MW",
        "Wind": "wind_MW",
        "Net Load": "net_load_MW",
        "Renewables": "renewables_MW",
    }
    keep = ["timestamp"]
    for old, new in prod_cols.items():
        prod[new] = pd.to_numeric(prod[old], errors="coerce")
        keep.append(new)
    prod5 = prod[keep].copy()
    return cur5, prod5


def prepare_caiso_15min(xlsx_path: str | Path, cfg: StudyConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    cur5, prod5 = load_caiso_workbook(xlsx_path)
    start = min(cur5.timestamp.min(), prod5.timestamp.min()).floor("D")
    end = max(cur5.timestamp.max(), prod5.timestamp.max()).ceil("D") - pd.Timedelta(minutes=5)
    idx = pd.date_range(start, end, freq="5min")
    base = pd.DataFrame({"timestamp": idx})
    base = base.merge(cur5, on="timestamp", how="left").merge(prod5, on="timestamp", how="left")
    for c in ["wind_curtailment_MW", "solar_curtailment_MW", "curtailment_MW", "n_reason_records"]:
        base[c] = base[c].fillna(0.0)

    base = base.set_index("timestamp")
    # Mean MW over each 15-min slot preserves energy when multiplied by 0.25 h.
    out = base.resample(f"{cfg.slot_minutes}min").mean(numeric_only=True)
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

    selected: List[Tuple[pd.Timestamp, pd.Timestamp, float]] = []
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
            "event_id": event_id,
            "start": st,
            "end": en,
            "raw_curtailment_MWh": score,
            "mean_curtailment_MW": float(sub.curtailment_MW.mean()),
            "max_curtailment_MW": float(sub.curtailment_MW.max()),
            "min_curtailment_MW": float(sub.curtailment_MW.min()),
            "fraction_positive": float((sub.curtailment_MW > 0).mean()),
        })
    return pd.DataFrame(rows)


def build_event_profiles(caiso15: pd.DataFrame, events: pd.DataFrame,
                         cfg: StudyConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    df = caiso15.set_index("timestamp")
    # One global scale factor preserves the relative severity of all selected CAISO events.
    selected_max = max(float(df.loc[r.start:r.end, "curtailment_MW"].max()) for r in events.itertuples())
    scale_kw_per_MW = cfg.pod_cap_kw / selected_max if selected_max > 0 else 0.0

    rows = []
    for r in events.itertuples():
        sub = df.loc[r.start:r.end].copy()
        if len(sub) != cfg.T:
            raise RuntimeError(f"Event {r.event_id} has {len(sub)} slots, expected {cfg.T}")
        allocated_kw = sub["curtailment_MW"].to_numpy(float) * scale_kw_per_MW
        target_kw = cfg.target_fraction_of_envelope * allocated_kw
        for slot, (ts, raw_mw, a_kw, t_kw) in enumerate(zip(sub.index, sub.curtailment_MW, allocated_kw, target_kw)):
            rows.append({
                "event_id": int(r.event_id),
                "slot": slot,
                "timestamp": ts,
                "raw_curtailment_MW": float(raw_mw),
                "allocated_curtailment_kW": float(a_kw),
                "dispatch_target_kW": float(t_kw),
                "global_scale_kW_per_MW": float(scale_kw_per_MW),
            })
    return pd.DataFrame(rows)


def load_alibaba_spot_jobs(job_csv: str | Path, cfg: StudyConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    jobs = pd.read_csv(job_csv)
    jobs = jobs[(jobs["job_type"] == "Spot") & (jobs["gpu_model"] == cfg.gpu_model)].copy()
    jobs["total_gpus"] = pd.to_numeric(jobs["gpu_request"], errors="coerce") * pd.to_numeric(jobs["worker_num"], errors="coerce")
    jobs["duration_minutes"] = pd.to_numeric(jobs["duration"], errors="coerce") / 60.0
    jobs = jobs[
        jobs["total_gpus"].between(8, 256)
        & jobs["duration_minutes"].between(cfg.job_duration_min_minutes, cfg.job_duration_max_minutes)
        & np.isclose(jobs["total_gpus"], np.round(jobs["total_gpus"]))
    ].copy()
    jobs["total_gpus"] = jobs["total_gpus"].round().astype(int)
    jobs["duration_slots"] = np.ceil(jobs["duration_minutes"] / cfg.slot_minutes).astype(int)
    jobs["gpu_power_kW"] = cfg.gpu_tdp_kw * jobs["total_gpus"]
    jobs["rounded_duration_minutes"] = jobs["duration_slots"] * cfg.slot_minutes
    jobs["start_options"] = cfg.T - jobs["duration_slots"] + 1
    jobs = jobs[jobs["start_options"] > 0].copy()
    return jobs


def select_fixed_job_portfolio(filtered_jobs: pd.DataFrame,
                               cfg: StudyConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    bins = [
        (8, 16, cfg.n_jobs_small),
        (17, 64, cfg.n_jobs_medium),
        (65, 256, cfg.n_jobs_large),
    ]
    parts = []
    for i, (lo, hi, n) in enumerate(bins):
        sub = filtered_jobs[filtered_jobs.total_gpus.between(lo, hi)]
        if len(sub) < n:
            raise RuntimeError(f"Not enough jobs in GPU bin {lo}-{hi}: have {len(sub)}, need {n}")
        parts.append(sub.sample(n=n, random_state=cfg.seed_jobs + i, replace=False))
    out = pd.concat(parts, ignore_index=True)
    out.insert(0, "portfolio_job_id", np.arange(len(out), dtype=int))
    keep = [
        "portfolio_job_id", "job_name", "organization", "gpu_model", "gpu_request",
        "worker_num", "total_gpus", "cpu_request", "submit_time", "duration",
        "duration_minutes", "duration_slots", "rounded_duration_minutes", "gpu_power_kW",
        "start_options", "job_type"
    ]
    return out[keep].sort_values("portfolio_job_id").reset_index(drop=True)


def build_start_variable_table(portfolio: pd.DataFrame, cfg: StudyConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    rows = []
    v = 0
    for r in portfolio.itertuples():
        L = int(r.duration_slots)
        for s in range(cfg.T - L + 1):
            rows.append({
                "var_index": v,
                "var_name": f"x_j{int(r.portfolio_job_id):02d}_s{s:02d}",
                "portfolio_job_id": int(r.portfolio_job_id),
                "start_slot": s,
                "end_slot_exclusive": s + L,
                "duration_slots": L,
                "total_gpus": int(r.total_gpus),
                "gpu_power_kW": float(r.gpu_power_kW),
            })
            v += 1
    return pd.DataFrame(rows)


def _add_linear(d: Dict[int, float], i: int, value: float) -> None:
    d[i] = d.get(i, 0.0) + float(value)


def _add_quad(d: Dict[Tuple[int, int], float], i: int, j: int, value: float) -> None:
    if i == j:
        raise ValueError("Quadratic dictionary expects off-diagonal pairs only")
    if i > j:
        i, j = j, i
    d[(i, j)] = d.get((i, j), 0.0) + float(value)


def build_qubo(portfolio: pd.DataFrame, event_profile: pd.DataFrame,
               cfg: StudyConfig = DEFAULT_CONFIG) -> Dict[str, Any]:
    """Build the unconstrained QUBO used by CPLEX-QUBO, SA and later LeapHybridBQMSampler.

    H = sum_t [(P_GPU(t)-P_target(t))/P_pod]^2
        + lambda_once * sum_j sum_{s<s'} x_{j,s} x_{j,s'}.

    The first term is genuinely quadratic because different job-start choices overlap in time.
    The second term enforces that a real job cannot be launched twice. Since the tracking term
    is normalized to O(T), lambda_once=30 is deliberately larger than the six-hour horizon T=24.
    """
    ep = event_profile.sort_values("slot")
    if len(ep) != cfg.T:
        raise ValueError(f"Expected {cfg.T} event slots, got {len(ep)}")

    var_table = build_start_variable_table(portfolio, cfg)
    n = len(var_table)
    linear: Dict[int, float] = {}
    quadratic: Dict[Tuple[int, int], float] = {}
    target = ep.dispatch_target_kW.to_numpy(float)
    P0 = float(cfg.pod_cap_kw)
    offset = float(np.sum((target / P0) ** 2))

    # Active variable lists by time slot.
    active_by_t: List[List[Tuple[int, float]]] = [[] for _ in range(cfg.T)]
    for r in var_table.itertuples():
        a = float(r.gpu_power_kW) / P0
        for t in range(int(r.start_slot), int(r.end_slot_exclusive)):
            active_by_t[t].append((int(r.var_index), a))

    # Tracking objective expansion.
    for t in range(cfg.T):
        rt = float(target[t] / P0)
        active = active_by_t[t]
        for v, a in active:
            # x^2=x for binary x.
            _add_linear(linear, v, a * a - 2.0 * rt * a)
        for i in range(len(active)):
            vi, ai = active[i]
            for j in range(i + 1, len(active)):
                vj, aj = active[j]
                _add_quad(quadratic, vi, vj, 2.0 * ai * aj)

    # At-most-once start penalty per empirical job.
    for job_id, g in var_table.groupby("portfolio_job_id"):
        vs = g.var_index.astype(int).tolist()
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                _add_quad(quadratic, vs[i], vs[j], cfg.lambda_once)

    # Drop tiny numerical zeros.
    linear = {i: v for i, v in linear.items() if abs(v) > 1e-15}
    quadratic = {ij: v for ij, v in quadratic.items() if abs(v) > 1e-15}

    return {
        "linear": linear,
        "quadratic": quadratic,
        "offset": offset,
        "var_table": var_table,
        "n_variables": n,
        "n_edges": len(quadratic),
        "density": len(quadratic) / (n * (n - 1) / 2) if n > 1 else 0.0,
        "event_id": int(ep.event_id.iloc[0]),
    }


def qubo_energy(sample: np.ndarray, qubo: Dict[str, Any], include_offset: bool = True) -> float:
    x = np.asarray(sample, dtype=float)
    e = float(qubo["offset"] if include_offset else 0.0)
    for i, b in qubo["linear"].items():
        e += b * x[i]
    for (i, j), q in qubo["quadratic"].items():
        e += q * x[i] * x[j]
    return float(e)


def decode_sample(sample: np.ndarray, portfolio: pd.DataFrame, event_profile: pd.DataFrame,
                  qubo: Dict[str, Any], cfg: StudyConfig = DEFAULT_CONFIG) -> Dict[str, Any]:
    x = np.asarray(sample, dtype=int)
    vt = qubo["var_table"]
    chosen = vt.loc[x[vt.var_index.to_numpy()] > 0].copy()
    load = np.zeros(cfg.T, dtype=float)
    for r in chosen.itertuples():
        load[int(r.start_slot):int(r.end_slot_exclusive)] += float(r.gpu_power_kW)

    counts = chosen.groupby("portfolio_job_id").size() if len(chosen) else pd.Series(dtype=int)
    duplicate_starts = int(np.sum(np.maximum(counts.to_numpy() - 1, 0))) if len(counts) else 0
    ep = event_profile.sort_values("slot")
    target = ep.dispatch_target_kW.to_numpy(float)
    envelope = ep.allocated_curtailment_kW.to_numpy(float)
    tracking = float(np.sum(((load - target) / cfg.pod_cap_kw) ** 2))
    penalty = float(cfg.lambda_once * sum(math.comb(int(c), 2) for c in counts if c >= 2))
    direct = tracking + penalty

    dt_h = cfg.slot_minutes / 60.0
    absorbed = np.minimum(load, envelope)
    metrics = {
        "objective_direct": direct,
        "tracking_objective": tracking,
        "once_penalty": penalty,
        "duplicate_starts": duplicate_starts,
        "n_jobs_scheduled": int((counts > 0).sum()) if len(counts) else 0,
        "tracking_RMSE_kW": float(np.sqrt(np.mean((load - target) ** 2))),
        "max_envelope_violation_kW": float(np.maximum(load - envelope, 0.0).max()),
        "envelope_violation_kWh": float(np.maximum(load - envelope, 0.0).sum() * dt_h),
        "allocated_curtailment_kWh": float(envelope.sum() * dt_h),
        "scheduled_gpu_energy_kWh": float(load.sum() * dt_h),
        "renewable_absorbed_kWh": float(absorbed.sum() * dt_h),
        "capture_fraction": float(absorbed.sum() / envelope.sum()) if envelope.sum() > 0 else 0.0,
        "target_energy_kWh": float(target.sum() * dt_h),
    }
    return {"chosen": chosen, "load_kW": load, "metrics": metrics}


def random_equivalence_tests(qubo: Dict[str, Any], portfolio: pd.DataFrame,
                             event_profile: pd.DataFrame, cfg: StudyConfig = DEFAULT_CONFIG,
                             n_tests: int = 20, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    n = int(qubo["n_variables"])
    for k in range(n_tests):
        # Sparse random vectors are enough to exercise both tracking and job-conflict couplings.
        x = (rng.random(n) < 0.06).astype(np.int8)
        dec = decode_sample(x, portfolio, event_profile, qubo, cfg)
        e_q = qubo_energy(x, qubo, include_offset=True)
        e_d = dec["metrics"]["objective_direct"]
        rows.append({"test": k, "direct": e_d, "qubo": e_q, "abs_error": abs(e_d - e_q)})
    return pd.DataFrame(rows)


def build_dimod_bqm(qubo: Dict[str, Any]):
    try:
        import dimod
    except Exception as e:
        raise ImportError("dimod is required. Install with: pip install dimod dwave-neal") from e
    return dimod.BinaryQuadraticModel(
        qubo["linear"], qubo["quadratic"], qubo["offset"], dimod.BINARY
    )


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
    return {"sample": sample, "energy": float(best.energy), "wall_time_s": wall, "sampleset": ss}


def solve_cplex_qubo(qubo: Dict[str, Any], cfg: StudyConfig = DEFAULT_CONFIG,
                     time_limit_s: float | None = None, log_output: bool = False) -> Dict[str, Any]:
    try:
        from docplex.mp.model import Model
    except Exception as e:
        raise ImportError("docplex + IBM CPLEX Python API are required. Install in your CPLEX environment.") from e

    mdl = Model(name=f"spot_gpu_qubo_event_{qubo['event_id']}")
    x = mdl.binary_var_list(qubo["n_variables"], name="x")
    # Generators avoid materializing a second 160k-term Python list.
    expr_lin = mdl.sum(coef * x[i] for i, coef in qubo["linear"].items())
    expr_quad = mdl.sum(coef * x[i] * x[j] for (i, j), coef in qubo["quadratic"].items())
    mdl.minimize(expr_lin + expr_quad + qubo["offset"])
    mdl.parameters.timelimit = float(time_limit_s or cfg.cplex_time_limit_s)
    mdl.parameters.mip.tolerances.mipgap = float(cfg.cplex_mip_gap)

    t0 = time.time()
    sol = mdl.solve(log_output=log_output)
    wall = time.time() - t0
    if sol is None:
        return {"sample": None, "energy": np.nan, "wall_time_s": wall, "status": str(mdl.solve_details.status),
                "best_bound": np.nan, "mip_gap": np.nan}
    sample = np.array([int(round(sol.get_value(v))) for v in x], dtype=np.int8)
    details = mdl.solve_details
    return {
        "sample": sample,
        "energy": float(sol.objective_value),
        "wall_time_s": wall,
        "status": str(details.status),
        "best_bound": float(getattr(details, "best_bound", np.nan)),
        "mip_gap": float(getattr(details, "mip_relative_gap", np.nan)),
    }


def solve_cplex_constrained_reference(portfolio: pd.DataFrame, event_profile: pd.DataFrame,
                                      cfg: StudyConfig = DEFAULT_CONFIG,
                                      time_limit_s: float | None = None,
                                      log_output: bool = False) -> Dict[str, Any]:
    """Hard-constrained MIQP reference: same start variables and squared tracking objective.

    It enforces at-most-one start per job and P_GPU(t) <= allocated curtailment(t).
    This is not the D-Wave QUBO; it is a physical reference used to validate that the 20% target
    headroom is sufficient before spending remote hybrid quota.
    """
    try:
        from docplex.mp.model import Model
    except Exception as e:
        raise ImportError("docplex + IBM CPLEX Python API are required.") from e

    vt = build_start_variable_table(portfolio, cfg)
    ep = event_profile.sort_values("slot")
    mdl = Model(name=f"spot_gpu_constrained_event_{int(ep.event_id.iloc[0])}")
    x = mdl.binary_var_list(len(vt), name="x")

    # Per-job uniqueness.
    for job_id, g in vt.groupby("portfolio_job_id"):
        inds = g.var_index.astype(int).tolist()
        mdl.add_constraint(mdl.sum(x[i] for i in inds) <= 1)

    target = ep.dispatch_target_kW.to_numpy(float)
    envelope = ep.allocated_curtailment_kW.to_numpy(float)
    loads = []
    for t in range(cfg.T):
        inds = vt[(vt.start_slot <= t) & (vt.end_slot_exclusive > t)]
        expr = mdl.sum(float(r.gpu_power_kW) * x[int(r.var_index)] for r in inds.itertuples())
        loads.append(expr)
        mdl.add_constraint(expr <= float(envelope[t]))

    obj = mdl.sum(((loads[t] - float(target[t])) / cfg.pod_cap_kw) ** 2 for t in range(cfg.T))
    mdl.minimize(obj)
    mdl.parameters.timelimit = float(time_limit_s or cfg.cplex_time_limit_s)
    mdl.parameters.mip.tolerances.mipgap = float(cfg.cplex_mip_gap)

    t0 = time.time()
    sol = mdl.solve(log_output=log_output)
    wall = time.time() - t0
    if sol is None:
        return {"sample": None, "energy": np.nan, "wall_time_s": wall, "status": str(mdl.solve_details.status),
                "best_bound": np.nan, "mip_gap": np.nan}
    sample = np.array([int(round(sol.get_value(v))) for v in x], dtype=np.int8)
    details = mdl.solve_details
    return {
        "sample": sample,
        "energy": float(sol.objective_value),
        "wall_time_s": wall,
        "status": str(details.status),
        "best_bound": float(getattr(details, "best_bound", np.nan)),
        "mip_gap": float(getattr(details, "mip_relative_gap", np.nan)),
    }


def prepare_all(caiso_xlsx: str | Path, cluster_zip: str | Path, output_dir: str | Path,
                cfg: StudyConfig = DEFAULT_CONFIG) -> Dict[str, Path]:
    out = _mkdir(Path(output_dir))
    ali_dir = _mkdir(out / "alibaba_extract")
    job_csv, node_csv, ali_readme = extract_spot_trace(cluster_zip, ali_dir)

    caiso15 = prepare_caiso_15min(caiso_xlsx, cfg)
    events = select_caiso_events(caiso15, cfg)
    profiles = build_event_profiles(caiso15, events, cfg)
    jobs = load_alibaba_spot_jobs(job_csv, cfg)
    portfolio = select_fixed_job_portfolio(jobs, cfg)
    vartable = build_start_variable_table(portfolio, cfg)

    caiso15_path = out / "caiso_15min.csv"
    events_path = out / "selected_caiso_events.csv"
    profiles_path = out / "event_profiles_long.csv"
    portfolio_path = out / "benchmark_spot_jobs.csv"
    vartable_path = out / "start_variable_table.csv"
    caiso15.to_csv(caiso15_path, index=False)
    events.to_csv(events_path, index=False)
    profiles.to_csv(profiles_path, index=False)
    portfolio.to_csv(portfolio_path, index=False)
    vartable.to_csv(vartable_path, index=False)

    # Build one BQM to record structural size; J is common to every event because target only changes h.
    q0 = build_qubo(portfolio, profiles[profiles.event_id == 0], cfg)
    eq0 = random_equivalence_tests(q0, portfolio, profiles[profiles.event_id == 0], cfg, n_tests=10)
    eq_path = out / "qubo_equivalence_smoketest.csv"
    eq0.to_csv(eq_path, index=False)

    manifest = {
        "config": asdict(cfg),
        "input_caiso": str(Path(caiso_xlsx).resolve()),
        "input_alibaba_zip": str(Path(cluster_zip).resolve()),
        "alibaba_spot_jobs_total_raw": int((pd.read_csv(job_csv, usecols=["job_type"]).job_type == "Spot").sum()),
        "filtered_A100_spot_jobs": int(len(jobs)),
        "selected_jobs": int(len(portfolio)),
        "selected_job_total_gpu_power_kW": float(portfolio.gpu_power_kW.sum()),
        "qubo_variables": int(q0["n_variables"]),
        "qubo_edges": int(q0["n_edges"]),
        "qubo_density": float(q0["density"]),
        "equivalence_max_abs_error": float(eq0.abs_error.max()),
        "gpu_power_source": "NVIDIA A100 80GB SXM standard TDP 400 W",
        "gpu_power_source_url": "https://www.nvidia.com/en-us/data-center/a100/",
        "alibaba_trace_source": "cluster-trace-v2026-spot-gpu",
        "caiso_note": "CAISO Production and Curtailments Data 2025; five-minute raw MW data; workbook covers Jan-May 2025.",
    }
    manifest_path = out / "data_and_qubo_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    return {
        "caiso15": caiso15_path,
        "events": events_path,
        "profiles": profiles_path,
        "portfolio": portfolio_path,
        "vartable": vartable_path,
        "equivalence": eq_path,
        "manifest": manifest_path,
        "alibaba_readme": ali_readme,
        "alibaba_nodes": node_csv,
    }
