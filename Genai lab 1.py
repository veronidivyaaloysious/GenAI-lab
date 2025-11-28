#!/usr/bin/env python3
"""
SLOWA - SLO Degradation Early-Warning Agent (local demo)
Save as slowa.py and run:

  python -m venv venv
  source venv/bin/activate    # or venv\Scripts\activate on Windows
  pip install pandas numpy matplotlib pydantic pyyaml scikit-learn
  python slowa.py

Outputs:
 - risk_report.json
 - forecast_<service>_<metric>.png (one per service+metric)
 - (optional) sample data created if missing

Notes:
 - No remote actions performed.
 - Ollama / LangChain integration points are placeholders only.
"""

from pathlib import Path
import json
from datetime import datetime, timedelta
import logging
import argparse
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ValidationError
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# --- Paths & defaults ---
BASE = Path.cwd() / "slowa_output"
METRICS_DIR = BASE / "metrics"
METRICS_CSV = METRICS_DIR / "metrics.csv"
SLO_YAML = BASE / "slo_targets.yml"
TOPO_JSON = BASE / "topology.json"
REPORT_JSON = BASE / "risk_report.json"

# --- Config models ---
class SLOTargetModel(BaseModel):
    latency_ms: float
    error_rate: float

class SLOConfigModel(BaseModel):
    slo_version: str
    targets: Dict[str, Dict[str, float]]

# --- Utilities: create sample data if missing ---
SAMPLE_METRICS = """timestamp,service,latency_ms,error_rate
2025-11-01T00:00:00,auth,120,0.002
2025-11-01T01:00:00,auth,130,0.001
2025-11-01T02:00:00,auth,140,0.001
2025-11-01T03:00:00,auth,150,0.002
2025-11-01T04:00:00,auth,210,0.010
2025-11-01T05:00:00,auth,230,0.015
2025-11-01T06:00:00,auth,250,0.020
2025-11-01T07:00:00,auth,240,0.018
2025-11-01T08:00:00,auth,220,0.012
2025-11-01T09:00:00,auth,200,0.009
2025-11-01T00:00:00,payment,90,0.0005
2025-11-01T01:00:00,payment,95,0.0006
2025-11-01T02:00:00,payment,100,0.0007
2025-11-01T03:00:00,payment,110,0.0008
2025-11-01T04:00:00,payment,115,0.0009
2025-11-01T05:00:00,payment,130,0.0015
2025-11-01T06:00:00,payment,140,0.0020
2025-11-01T07:00:00,payment,150,0.0030
2025-11-01T08:00:00,payment,160,0.0040
2025-11-01T09:00:00,payment,170,0.0050
"""

SAMPLE_SLO = {
    "slo_version": "1",
    "targets": {
        "auth": {"latency_ms": 200.0, "error_rate": 0.01},
        "payment": {"latency_ms": 150.0, "error_rate": 0.002}
    }
}

SAMPLE_TOPO = {
    "services": {
        "auth": {
            "components": ["auth-api", "auth-db"],
            "max_replicas": 10,
            "min_replicas": 2
        },
        "payment": {
            "components": ["payment-api", "payment-db", "queue"],
            "max_replicas": 8,
            "min_replicas": 1
        }
    },
    "infrastructure": {
        "cluster": "local-cluster",
        "autoscaler_enabled": True
    }
}

def ensure_sample_files():
    BASE.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    if not METRICS_CSV.exists():
        METRICS_CSV.write_text(SAMPLE_METRICS)
        logging.info(f"Created sample metrics CSV at {METRICS_CSV}")
    if not SLO_YAML.exists():
        with open(SLO_YAML, "w") as f:
            yaml.safe_dump(SAMPLE_SLO, f)
        logging.info(f"Created sample slo_targets.yml at {SLO_YAML}")
    if not TOPO_JSON.exists():
        with open(TOPO_JSON, "w") as f:
            json.dump(SAMPLE_TOPO, f, indent=2)
        logging.info(f"Created sample topology.json at {TOPO_JSON}")

# --- Loaders ---
def load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values(["service", "timestamp"]).reset_index(drop=True)
    return df

def load_slo(path: Path) -> Dict[str, Dict[str, float]]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    try:
        SLOConfigModel(**raw)
    except ValidationError as e:
        logging.warning("SLO config validation warning: %s", e)
    return raw.get("targets", {})

def load_topology(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)

# --- Drift detector (rolling z-score with scaling) ---
class DriftDetector:
    def __init__(self, window: int = 5, z_thresh: float = 2.0):
        self.window = window
        self.z_thresh = z_thresh

    def detect(self, series: pd.Series) -> pd.Series:
        # handles small windows robustly
        rolling_mean = series.rolling(self.window, min_periods=1).mean()
        rolling_std = series.rolling(self.window, min_periods=1).std().replace(0, 1.0).fillna(1.0)
        z = (series - rolling_mean) / rolling_std
        return z.abs() > self.z_thresh

# --- Forecaster: Exponential smoothing with simple seasonality detection ---
class SimpleForecaster:
    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha

    def fit_smooth(self, series: pd.Series):
        # Single exponential smoothing state
        if len(series) == 0:
            return np.nan
        s = series.iloc[0]
        for v in series.iloc[1:]:
            s = self.alpha * v + (1 - self.alpha) * s
        return float(s)

    def forecast(self, series: pd.Series, steps: int = 3) -> List[float]:
        state = self.fit_smooth(series)
        if np.isnan(state):
            return [np.nan] * steps
        # naive approach: repeat state for steps
        return [state for _ in range(steps)]

# --- Risk scoring ---
class RiskScorer:
    """
    Produces score 0..100 where >60 is considered imminent.
    Scoring formula is soft and saturates: transforms distance to SLO into a score.
    """
    def score(self, current: float, forecast_mean: float, slo_threshold: float) -> float:
        if np.isnan(forecast_mean):
            return 0.0
        distance = forecast_mean - slo_threshold
        # Only care when forecast exceeds the threshold
        if distance <= 0:
            return 0.0
        # Soft scaling using ratio
        ratio = distance / max(1e-6, slo_threshold)
        # map ratio (0..inf) -> score (0..100) with diminishing returns
        score = 100.0 * (1 - 1.0 / (1 + ratio))
        return float(min(100.0, max(0.0, score)))

# --- Mitigation proposer (rule-based + topology-aware) ---
class MitigationProposer:
    def __init__(self, topology: Dict[str, Any]):
        self.topology = topology

    def propose(self, service: str, metric: str, risk_score: float) -> List[Dict[str, Any]]:
        suggestions = []
        svc_info = self.topology.get("services", {}).get(service, {})
        autoscaler = self.topology.get("infrastructure", {}).get("autoscaler_enabled", False)

        # Low risk: monitor
        if risk_score < 30:
            suggestions.append({"action": "monitor", "reason": "Low risk; continue monitoring"})
            return suggestions

        # Medium / high risk: propose actions
        if metric == "latency_ms":
            if autoscaler:
                suggestions.append({
                    "action": "scale_up",
                    "reason": "Autoscaler available: increase replicas / CPU; consider temporary scale up",
                    "service": service
                })
            else:
                suggestions.append({
                    "action": "investigate_db",
                    "reason": "Check DB queries, caches, or CDN; reduce heavy operations if possible",
                    "service": service
                })
        elif metric == "error_rate":
            suggestions.append({
                "action": "traffic_control",
                "reason": "Apply rate limiting or circuit-breaker on downstream calls and fallback if possible",
                "service": service
            })
        # add common steps
        suggestions.append({
            "action": "increase_alert_priority",
            "reason": f"Risk {risk_score:.1f} - raise to on-call for manual triage",
            "service": service
        })
        # Must include human approval per guardrail
        suggestions.append({
            "action": "human_approval",
            "reason": "Per guardrails no automatic remote actions: require human approval before execute"
        })
        return suggestions

# --- Optional extension point: LLM-based mitigation explanation (placeholder) ---
def llm_explain_mitigation_placeholder(service: str, metric: str, context: Dict[str, Any]) -> str:
    """
    Placeholder function showing where you'd call a local Ollama/LLM to generate
    a human-readable explanation. THIS FUNCTION DOES NOT CALL ANY REMOTE SERVICE.
    Replace with an HTTP call to local Ollama if you have one (e.g., http://127.0.0.1:11434).
    """
    # Compose a short deterministic explanation (not using external LLM)
    return (
        f"Suggested mitigations for {service} on {metric}: "
        "1) Investigate resource saturation and slow DB queries. "
        "2) Consider scaling replicas if autoscaler is enabled. "
        "3) Activate rate limiting for noisy clients. "
        "All remediation actions require human approval."
    )

# --- Pipeline runner ---
def run_pipeline(window: int = 5, z_thresh: float = 2.0, forecast_steps: int = 3, alpha: float = 0.4):
    logging.info("Starting SLOWA pipeline")
    ensure_sample_files()
    df = load_metrics(METRICS_CSV)
    slo_targets = load_slo(SLO_YAML)
    topology = load_topology(TOPO_JSON)

    dd = DriftDetector(window=window, z_thresh=z_thresh)
    forecaster = SimpleForecaster(alpha=alpha)
    scorer = RiskScorer()
    proposer = MitigationProposer(topology)

    entries = []
    # analyze per service and metric
    for service, group in df.groupby("service"):
        group = group.sort_values("timestamp")
        timestamps = pd.to_datetime(group["timestamp"]).tolist()
        for metric in ["latency_ms", "error_rate"]:
            series = group[metric].reset_index(drop=True).astype(float)
            if len(series) == 0:
                continue
            # drift detection
            drift_series = dd.detect(series)
            recent_drift = bool(drift_series.iloc[-1])

            # forecast
            forecasts = forecaster.forecast(series, steps=forecast_steps)
            forecast_mean = float(np.nanmean(forecasts)) if len(forecasts) else float("nan")

            # SLO threshold
            slo_threshold = slo_targets.get(service, {}).get(metric)
            if slo_threshold is None:
                logging.debug("No SLO threshold for %s %s; skipping", service, metric)
                continue

            # risk
            risk_score = scorer.score(current=float(series.iloc[-1]), forecast_mean=forecast_mean, slo_threshold=slo_threshold)

            # mitigation proposals
            mitigations = proposer.propose(service, metric, risk_score)

            # human-friendly explanation via placeholder LLM (deterministic)
            explanation = llm_explain_mitigation_placeholder(service, metric, {
                "latest": float(series.iloc[-1]),
                "forecast_mean": forecast_mean,
                "slo_threshold": slo_threshold,
                "risk_score": risk_score
            })

            entry = {
                "service": service,
                "metric": metric,
                "latest": float(series.iloc[-1]),
                "forecast_next_mean": forecast_mean,
                "slo_threshold": slo_threshold,
                "risk_score": risk_score,
                "drift_recent": recent_drift,
                "mitigations": mitigations,
                "mitigation_explanation": explanation
            }
            entries.append(entry)

            # produce forecast chart
            try:
                # compute future timestamps evenly spaced by last interval
                if len(timestamps) >= 2:
                    last_delta = pd.to_datetime(timestamps[-1]) - pd.to_datetime(timestamps[-2])
                else:
                    last_delta = pd.Timedelta(hours=1)
                future_ts = [pd.to_datetime(timestamps[-1]) + (i + 1) * last_delta for i in range(forecast_steps)]
                plt.figure(figsize=(8, 3.5))
                plt.plot(pd.to_datetime(timestamps), series, marker="o", label="observed")
                plt.plot(future_ts, forecasts, marker="x", linestyle="--", label="forecast")
                plt.axhline(slo_threshold, color="red", linewidth=1, label="SLO threshold")
                plt.title(f"{service} - {metric} (risk={risk_score:.1f})")
                plt.xlabel("timestamp")
                plt.ylabel(metric)
                plt.legend()
                plt.tight_layout()
                chart_path = BASE / f"forecast_{service}_{metric}.png"
                plt.savefig(chart_path)
                plt.close()
                logging.info("Wrote chart: %s", chart_path)
            except Exception as e:
                logging.warning("Failed to write chart for %s %s: %s", service, metric, e)

    # summary and success metric scaffolding
    imminent = [e for e in entries if e["risk_score"] > 60.0]
    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "n_entries": len(entries),
        "n_imminent": len(imminent),
        "imminent_services": [f"{e['service']}:{e['metric']}" for e in imminent],
        "entries": entries
    }

    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    logging.info("Wrote report: %s", REPORT_JSON)
    logging.info("SLOWA pipeline finished")

# --- CLI ---
def parse_args():
    p = argparse.ArgumentParser(description="Run local SLOWA (SLO Degradation Early-Warning Agent)")
    p.add_argument("--window", type=int, default=5, help="rolling window for drift detector")
    p.add_argument("--z_thresh", type=float, default=2.0, help="z-score threshold for drift")
    p.add_argument("--steps", type=int, default=3, help="forecast steps")
    p.add_argument("--alpha", type=float, default=0.4, help="smoothing alpha for forecaster")
    p.add_argument("--no-sample", action="store_true", help="do not create sample files if missing")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.no_sample:
        # ensure directory exists but don't auto-create sample content
        BASE.mkdir(parents=True, exist_ok=True)
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
    # run pipeline (creates sample files by default)
    run_pipeline(window=args.window, z_thresh=args.z_thresh, forecast_steps=args.steps, alpha=args.alpha)
    print(f"Done. Outputs in {BASE}. Review {REPORT_JSON} and forecast_*.png")
