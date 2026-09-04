"""D-Wave Leap hybrid BQM utilities for the field-calibrated AI-curtailment QUBO.

The validated physical/QUBO formulation stays in ``ai_spot_curtailment_dvfs_qubo_core.py``.
This module adds only remote Leap-hybrid submission, metadata capture, and robust
per-event checkpoint/resume support. It does not alter any QUBO coefficient.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import hashlib
import json
import time

import numpy as np
import pandas as pd

from ai_spot_curtailment_dvfs_qubo_core import build_dimod_bqm, qubo_energy, decode_sample

try:
    from dwave.system import LeapHybridBQMSampler as _HybridSampler
    HYBRID_CLASS_NAME = "LeapHybridBQMSampler"
except Exception:
    try:
        from dwave.system import LeapHybridSampler as _HybridSampler
        HYBRID_CLASS_NAME = "LeapHybridSampler"
    except Exception:
        _HybridSampler = None
        HYBRID_CLASS_NAME = None


def json_safe(v: Any):
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {str(k): json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [json_safe(x) for x in v]
    try:
        return float(v)
    except Exception:
        return str(v)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def create_sampler(**kwargs):
    if _HybridSampler is None:
        raise RuntimeError(
            "D-Wave Ocean hybrid sampler is unavailable. Install dwave-ocean-sdk "
            "and configure Leap credentials before running the campaign."
        )
    return _HybridSampler(**kwargs)


def sampler_identity(sampler) -> dict:
    solver = getattr(sampler, 'solver', None)
    props = getattr(sampler, 'properties', {}) or {}
    return {
        'python_sampler_class': sampler.__class__.__name__,
        'imported_hybrid_class': HYBRID_CLASS_NAME,
        'solver_name': getattr(solver, 'name', None),
        'solver_id': getattr(solver, 'id', None),
        'quota_conversion_rate': props.get('quota_conversion_rate', np.nan),
        'maximum_number_of_variables': props.get('maximum_number_of_variables', np.nan),
        'maximum_number_of_biases': props.get('maximum_number_of_biases', np.nan),
        'minimum_time_limit_property': json_safe(props.get('minimum_time_limit', None)),
        'version': json_safe(props.get('version', None)),
    }


def min_time_limit_seconds(sampler, bqm) -> float:
    try:
        return float(sampler.min_time_limit(bqm))
    except Exception:
        return float('nan')


def _timing_value(info: dict, key: str):
    """Read hybrid timing fields from either top-level info or nested timing dict."""
    if key in info:
        return info.get(key)
    timing = info.get('timing', {})
    if isinstance(timing, dict) and key in timing:
        return timing.get(key)
    return None


def parse_hybrid_timing(info: dict) -> dict:
    """Preserve raw timing values and provide seconds assuming Ocean hybrid µs units."""
    out = {}
    for key in ('charge_time', 'run_time', 'qpu_access_time'):
        raw = _timing_value(info, key)
        out[f'{key}_raw'] = raw
        try:
            out[f'{key}_s'] = float(raw) / 1e6
        except Exception:
            out[f'{key}_s'] = np.nan
    return out


def solve_one_event(qubo: dict, *, sampler, time_limit: float = 30.0,
                    label_prefix: str = 'AI-curtailment-DVFS') -> tuple[np.ndarray, dict]:
    bqm = build_dimod_bqm(qubo)
    min_t = min_time_limit_seconds(sampler, bqm)
    if np.isfinite(min_t) and float(time_limit) + 1e-12 < min_t:
        raise ValueError(f'time_limit={time_limit:g}s is below solver minimum {min_t:g}s')

    eid = int(qubo['event_id'])
    label = f'{label_prefix}-e{eid:02d}-n{qubo["n_variables"]}'
    t0 = time.perf_counter()
    ss = sampler.sample(bqm, time_limit=float(time_limit), label=label)
    best = ss.first  # force remote completion before stopping wall-clock timer
    wall = time.perf_counter() - t0

    sample = np.array([int(best.sample[i]) for i in range(int(qubo['n_variables']))], dtype=np.int8)
    recomputed = qubo_energy(sample, qubo, include_offset=True)
    info = dict(getattr(ss, 'info', {}) or {})
    timing = parse_hybrid_timing(info)
    solver = getattr(sampler, 'solver', None)

    meta = {
        'event_id': eid,
        'problem_label': label,
        'problem_id': info.get('problem_id', info.get('id', None)),
        'sampler_class': sampler.__class__.__name__,
        'solver_name': getattr(solver, 'name', None),
        'solver_id': getattr(solver, 'id', None),
        'requested_time_limit_s': float(time_limit),
        'minimum_time_limit_s': min_t,
        'client_wall_time_s': float(wall),
        'solver_energy_reported': float(best.energy),
        'qubo_energy_recomputed': float(recomputed),
        'energy_abs_difference': float(abs(float(best.energy) - float(recomputed))),
        **timing,
        'sampleset_info': json_safe(info),
    }
    return sample, meta


def _atomic_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(json_safe(obj), indent=2))
    tmp.replace(path)


def _atomic_csv(path: Path, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def build_campaign_fingerprint(data_dir: str | Path, core_py: str | Path,
                               time_limit: float, event_ids: Iterable[int]) -> dict:
    data_dir = Path(data_dir)
    files = [
        data_dir/'benchmark_spot_jobs_dvfs.csv',
        data_dir/'emerald_dvfs_calibration.csv',
        data_dir/'event_profiles_long.csv',
        data_dir/'revised_data_and_qubo_manifest.json',
        Path(core_py),
    ]
    return {
        'time_limit_s': float(time_limit),
        'event_ids': [int(x) for x in event_ids],
        'sha256': {p.name: sha256_file(p) for p in files},
    }


def initialize_or_validate_campaign(output_dir: str | Path, fingerprint: dict,
                                    sampler_info: dict, *, resume: bool = True):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = out/'campaign_config.json'
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        if old.get('fingerprint') != json_safe(fingerprint):
            raise RuntimeError(
                'Existing D-Wave output belongs to a different dataset/QUBO/time-limit campaign. '
                'Use a new output directory rather than mixing runs.'
            )
        if not resume:
            raise RuntimeError('Output directory already contains a campaign; RESUME=False.')
    else:
        _atomic_json(cfg_path, {'fingerprint': fingerprint, 'sampler': sampler_info})
    return out


def event_is_complete(output_dir: str | Path, event_id: int) -> bool:
    e = Path(output_dir)/'events'/f'event_{int(event_id):02d}'
    return (e/'result.npz').exists() and (e/'metadata.json').exists() and (e/'chosen_variables.csv').exists()


def save_event_result(output_dir: str | Path, event_id: int, sample: np.ndarray,
                      decoded: dict, metadata: dict, event_profile: pd.DataFrame):
    e = Path(output_dir)/'events'/f'event_{int(event_id):02d}'
    e.mkdir(parents=True, exist_ok=True)
    # Write NPZ atomically.
    tmp = e/'result.tmp.npz'
    np.savez_compressed(
        tmp,
        sample=np.asarray(sample, np.int8),
        load_kW=np.asarray(decoded['load_kW'], float),
        gpu_count=np.asarray(decoded['gpu_count'], float),
    )
    tmp.replace(e/'result.npz')
    decoded['chosen'].to_csv(e/'chosen_variables.csv', index=False)
    traj = event_profile.sort_values('slot').copy()
    traj['dwave_gpu_load_kW'] = decoded['load_kW']
    traj['dwave_active_gpus'] = decoded['gpu_count']
    traj.to_csv(e/'trajectory.csv', index=False)
    _atomic_json(e/'metadata.json', {'dwave': metadata, 'metrics': decoded['metrics']})


def rebuild_summary(output_dir: str | Path) -> pd.DataFrame:
    out = Path(output_dir)
    rows = []
    events_dir = out/'events'
    if events_dir.exists():
        for e in sorted(events_dir.glob('event_*')):
            meta_path = e/'metadata.json'
            if not meta_path.exists():
                continue
            obj = json.loads(meta_path.read_text())
            row = dict(obj.get('metrics', {}))
            row.update({k: v for k, v in obj.get('dwave', {}).items() if k != 'sampleset_info'})
            rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values('event_id').reset_index(drop=True)
    _atomic_csv(out/'dwave_hybrid_summary.csv', df)
    return df
