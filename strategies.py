"""Explicit discrete strategies, cash commitments, and investment restrictions."""
from __future__ import annotations

STRATEGIES = {
    "No new deployment": {"programmes": [], "priority": "Exit stewardship"},
    "IT Full": {"programmes": ["IT Full"], "priority": "IT Engineering"},
    "IT Full + Security Pilot": {"programmes": ["IT Full", "Security Pilot"], "priority": "IT Engineering"},
    "Security Pilot only": {"programmes": ["Security Pilot"], "priority": "Limited Security oversight"},
    "Security Full": {"programmes": ["Security Full"], "priority": "Security Hardware"},
    "E-commerce Full": {"programmes": ["E-commerce Full"], "priority": "E-commerce"},
    "E-commerce Partial": {"programmes": ["E-commerce Partial"], "priority": "Limited stabilization"},
    "IT Full + E-commerce Partial": {"programmes": ["IT Full", "E-commerce Partial"], "priority": "IT Engineering"},
    "Security Pilot + E-commerce Full": {"programmes": ["Security Pilot", "E-commerce Full"], "priority": "E-commerce"},
    "Security Pilot + E-commerce Partial": {"programmes": ["Security Pilot", "E-commerce Partial"], "priority": "Limited interventions"},
    "IT Full + Security Pilot + E-commerce Partial": {"programmes": ["IT Full", "Security Pilot", "E-commerce Partial"], "priority": "IT Engineering"},
    "IT Full (Staged)": {"programmes": ["IT Full"], "priority": "IT Engineering", "staged": True},
    "IT Full + Security Pilot (Staged)": {"programmes": ["IT Full", "Security Pilot"], "priority": "IT Engineering", "staged": True},
}

def funding_split(programme: str, cfg: dict) -> tuple[float,float]:
    p = cfg["programmes"][programme]
    cost = float(p["total_cost"])
    if cost < 0:
        raise ValueError("Programme cost must be non-negative")
    mode = cfg.get("financing_mode", "Fund only")
    if mode == "Pro-rata co-funding":
        if not cfg.get("cofund_confirmed", False):
            raise ValueError("Pro-rata shareholder participation is not confirmed. Enter explicit confirmation.")
        share = 1.0 - float(cfg["ownership"][p["company"]])
        if not 0 <= share <= 1:
            raise ValueError("Effective ownership outside [0,1]")
        return cost * (1-share), cost * share
    if mode == "Confirmed co-funding":
        if not cfg.get("cofund_confirmed", False):
            raise ValueError("Confirmed co-funding must be explicitly enabled.")
        paid=float(cfg.get("fund_contributions",{}).get(programme,cost))
        if not 0 <= paid <= cost:
            raise ValueError("Fund contribution must be within total programme cost")
        return paid, cost-paid
    if mode not in ("Fund only", "Staged fund-only"):
        raise ValueError("Financing terms unspecified: priced equity and shareholder loans are disabled")
    return cost, 0.0

def validate_strategy(name: str, cfg: dict) -> dict:
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}")
    strategy = STRATEGIES[name]
    names = strategy["programmes"]
    if sum(bool(cfg["programmes"][n]["intensive"]) for n in names) > 1:
        raise ValueError("Only one intensive management programme is permitted")
    companies = [cfg["programmes"][n]["company"] for n in names]
    if len(companies) != len(set(companies)):
        raise ValueError("Mutually exclusive programmes cannot be stacked on one company")
    funding = [funding_split(n, cfg) for n in names]
    deployed = sum(pair[0] for pair in funding)
    other = sum(pair[1] for pair in funding)
    budget = float(cfg["budget"])
    reserve = float(cfg.get("minimum_reserve", 0))
    if budget < 0 or reserve < 0 or reserve > budget:
        raise ValueError("Budget/reserve invalid")
    if deployed > budget - reserve + 1e-9:
        raise ValueError(f"Insufficient fund capital: strategy requires ${deployed:.2f}m; cap less reserve is ${budget-reserve:.2f}m")
    return {"deployed": deployed, "other_funding": other, "retained": budget-deployed, "priority": strategy["priority"],
            "staged": bool(strategy.get("staged", False)), "programmes": names}

def eligible_strategies(cfg: dict) -> dict[str, dict]:
    results = {}
    for name in STRATEGIES:
        try:
            results[name] = validate_strategy(name, cfg)
        except ValueError:
            continue
    return results
