"""Pure deterministic company valuation and fund-level terminal cash reconciliation."""
from __future__ import annotations
import copy
import json
from pathlib import Path
from typing import Any
import numpy as np
from strategies import validate_strategy

DATA = Path(__file__).resolve().parent / "data"
COMPANIES = json.loads((DATA / "companies.json").read_text())["companies"]
BASE = json.loads((DATA / "c9_base.json").read_text())
C8 = json.loads((DATA / "c8_scenarios.json").read_text())
NAMES = list(COMPANIES)

def default_config() -> dict:
    cfg = {
        "budget": float(BASE["budget"]), "minimum_reserve": 0.0,
        "no_action": copy.deepcopy(BASE["no_action"]),
        "programmes": copy.deepcopy(BASE["programmes"]),
        "ownership": {c: float(v["ownership"]) for c,v in COMPANIES.items()},
        "forecast_modes": {c:"Direct EBITDA" for c in NAMES},
        "forecast_margins": {c: BASE["no_action"][c]["ebitda"] / BASE["no_action"][c]["exit_revenue"] for c in NAMES},
        "financing_mode": "Fund only", "cofund_confirmed": False,
        "fund_contributions": {name:float(p["total_cost"]) for name,p in BASE["programmes"].items()},
        "cofund_shares": {c: 1-COMPANIES[c]["ownership"] for c in NAMES},
        "milestone_upfront_fraction": 0.5, "milestone_probability": 0.75,
        "fees": 0.0, "escrow": 0.0, "deferred": 0.0,
        "exit_months": {c: 24 for c in NAMES},
        "distressed_recovery": {c: 0.0 for c in NAMES},
        "contractual_receipts": {c: 0.0 for c in NAMES},
        "close_probabilities": {c: 1.0 for c in NAMES},
        "simulation": {
            "profile": "C8 paired", "n": 10000, "seed": 42,
            "market_probs": copy.deepcopy(C8["market_probabilities"]),
            "execution_probs": copy.deepcopy(C8["conditional_execution"]),
            "ebitda_vol": 0.0, "multiple_vol": 0.0, "debt_vol": 0.0,
            "common_factor": 0.5, "execution_shared_probability": 1.0,
            "closing_shared_probability": 1.0
        }
    }
    return cfg

def validate_config(cfg: dict) -> None:
    for label in ("fees","escrow","deferred"):
        v = float(cfg[label])
        if not 0 <= v <= 1:
            raise ValueError(f"{label}: must be 0% to 100%")
    if cfg["escrow"]+cfg["deferred"] > 1+1e-9:
        raise ValueError("Escrow and deferred portions cannot exceed 100%")
    if float(cfg["budget"]) < 0 or float(cfg["minimum_reserve"]) > float(cfg["budget"]):
        raise ValueError("Budget or minimum retained capital invalid")
    for c in NAMES:
        if not 0 <= float(cfg["ownership"][c]) <= 1:
            raise ValueError(f"Ownership invalid: {c}")
        if float(cfg["no_action"][c]["multiple"]) < 0:
            raise ValueError(f"Exit multiple invalid: {c}")
        if float(cfg["no_action"][c]["exit_revenue"]) < 0:
            raise ValueError(f"Exit revenue invalid: {c}")
        if float(cfg["no_action"][c]["debt"]) < 0:
            raise ValueError(f"Net debt invalid: {c}")
        if not 0 <= float(cfg["close_probabilities"][c]) <= 1:
            raise ValueError(f"Close probability invalid: {c}")
    for p in cfg["programmes"].values():
        if p["total_cost"] < 0 or p["funded_multiple"] < 0:
            raise ValueError("Funding or multiple cannot be negative")

def forecast_baseline(company: str, cfg: dict) -> dict:
    f = dict(cfg["no_action"][company])
    if cfg.get("forecast_modes", {}).get(company) == "Revenue × Margin":
        f["ebitda"] = f["exit_revenue"] * cfg["forecast_margins"][company]
    return f

def exit_equity(ebitda: Any, multiple: Any, debt: Any, distressed_recovery: Any=0.0):
    """EV/EBITDA is disabled for nonpositive EBITDA. Explicit recovery input is used instead."""
    e, m, d, r = np.broadcast_arrays(
        np.asarray(ebitda, dtype=float), np.asarray(multiple, dtype=float),
        np.asarray(debt, dtype=float), np.asarray(distressed_recovery, dtype=float))
    if np.any(m < 0) or np.any(d < 0) or np.any(r < 0):
        raise ValueError("Multiple, debt and recovery must not be negative")
    result = np.where(e > 0, np.maximum(0.0, e * m - d), r)
    return float(result) if result.ndim == 0 else result

def realised_cash(equity: Any, ownership: float, *, closing: Any=True, fees: float=0,
                  escrow: float=0, deferred: float=0, contractual: Any=0.0):
    """Returns (gross proceeds, cash received by M24, withheld/deferred, unclosed unrealized NAV)."""
    amount = np.asarray(equity) * ownership + np.asarray(contractual)
    net_after_fee = amount * (1-fees)
    close = np.asarray(closing, dtype=bool)
    received = np.where(close, net_after_fee * (1-escrow-deferred), 0.0)
    withheld = np.where(close, net_after_fee * (escrow+deferred), 0.0)
    unrealized = np.where(close, 0.0, net_after_fee)
    return amount, received, withheld, unrealized

def company_forecast(company: str, cfg: dict, programme: str | None=None, effectiveness: float=1.0):
    base = forecast_baseline(company, cfg)
    if programme is None:
        return base
    p = cfg["programmes"][programme]
    # Explicit documented driver: success operates as a delta to the editable unfunded exit forecast.
    original_baseline = BASE["no_action"][company]
    goal_ebitda_delta = p["funded_ebitda"] - original_baseline["ebitda"]
    premium = p["funded_multiple"] - original_baseline["multiple"]
    return {**base, "ebitda":base["ebitda"]+effectiveness*goal_ebitda_delta,
            "multiple":max(0.0,base["multiple"]+effectiveness*premium),
            "debt":max(0.0,base["debt"]+effectiveness*p.get("funded_debt_change",0.0))}

def evaluate(name: str, cfg: dict, *, effects: dict[str,float] | None=None,
             closes: dict[str,bool] | None=None, actual_deployed: float | None=None) -> dict:
    """Fund terminal cash = realized exit proceeds + budget less capital consumed, exactly once."""
    validate_config(cfg)
    info=validate_strategy(name,cfg)
    by_company={cfg["programmes"][p]["company"]:p for p in info["programmes"]}
    rows={}
    for company in NAMES:
        prog=by_company.get(company)
        level=1.0 if effects is None else effects.get(prog,1.0)
        f=company_forecast(company,cfg,prog,level)
        equity=exit_equity(f["ebitda"], f["multiple"], f["debt"], cfg["distressed_recovery"][company])
        paid= (cfg["exit_months"][company] <= 24) if closes is None else bool(closes[company] and cfg["exit_months"][company] <=24)
        gross, cash, withheld, unrealized = realised_cash(
            equity,cfg["ownership"][company],closing=paid, fees=cfg["fees"],
            escrow=cfg["escrow"], deferred=cfg["deferred"],
            contractual=cfg["contractual_receipts"][company])
        rows[company]={"programme":prog or "No action", **f, "equity":float(equity),
                       "gross":float(gross), "cash":float(cash),
                       "withheld":float(withheld),"unrealized":float(unrealized),
                       "closed":bool(paid)}
    spent=info["deployed"] if actual_deployed is None else float(actual_deployed)
    if spent < 0 or spent > cfg["budget"]+1e-9:
        raise ValueError("Actual funding violates fund budget")
    retained=cfg["budget"]-spent
    gross=sum(r["gross"] for r in rows.values())
    realised=sum(r["cash"] for r in rows.values())
    total=realised+retained
    return {"strategy":name, "gross":gross,"realized":realised,
            "deployed":spent,"other_shareholder_funding":info["other_funding"],
            "retained":retained,"terminal_cash":total,
            "withheld":sum(r["withheld"] for r in rows.values()),
            "unrealized":sum(r["unrealized"] for r in rows.values()),
            "all_closed":all(r["closed"] for r in rows.values()),
            "priority":info["priority"],"companies":rows}

def compare(name: str,cfg:dict) -> dict:
    chosen=evaluate(name,cfg)
    baseline=evaluate("No new deployment",cfg)
    chosen["no_action"]=baseline["terminal_cash"]
    chosen["incremental"]=chosen["terminal_cash"]-baseline["terminal_cash"]
    chosen["return_per_dollar"]=chosen["incremental"]/chosen["deployed"] if chosen["deployed"] else None
    return chosen
