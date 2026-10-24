"""Paired common-random-number Monte Carlo; illustrative assumptions, not calibrated forecasts."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from model import BASE, C8, NAMES, forecast_baseline, exit_equity, validate_config
from strategies import eligible_strategies, funding_split

STATES = ("Adverse", "Base", "Favorable")
EXECUTIONS = ("Success","Partial","Failure","Delay")

def validate_probabilities(probabilities, label="probabilities"):
    vals=np.asarray(probabilities,dtype=float)
    if vals.ndim != 1 or not np.all(np.isfinite(vals)) or np.any(vals < 0) or not np.isclose(vals.sum(),1,atol=1e-8):
        raise ValueError(f"{label} must contain non-negative values adding to 1")
    return vals

def validate_simulation(cfg):
    s=cfg["simulation"]
    validate_config(cfg)
    if s["profile"] not in ("C8 paired","C9 transaction base"):
        raise ValueError("Unknown simulation profile")
    if int(s["n"]) not in (1000,5000,10000,25000,100000):
        raise ValueError("Unsupported simulation size")
    validate_probabilities([s["market_probs"][k] for k in STATES],"market")
    for k in STATES:
        validate_probabilities([s["execution_probs"][k][e] for e in EXECUTIONS],f"execution[{k}]")
    for v in ("ebitda_vol","multiple_vol","debt_vol"):
        if float(s[v]) < 0 or not np.isfinite(s[v]):
            raise ValueError(f"{v} must be non-negative")
    for v in ("common_factor","execution_shared_probability","closing_shared_probability"):
        if not 0 <= s[v] <= 1:
            raise ValueError(f"{v} must be between 0 and 1")
    if not 0 <= cfg["milestone_probability"] <= 1 or not 0 <= cfg["milestone_upfront_fraction"] <= 1:
        raise ValueError("Staged milestone controls invalid")

def draw_categorical(rng, probabilities_by_state, state_indices, n, *, shared_probability=1, common_uniform=None):
    """Categorical execution sampling conditional on common market state."""
    if common_uniform is None:
        common_uniform=rng.random(n)
    u=np.where(rng.random(n) < shared_probability,common_uniform,rng.random(n))
    result=np.zeros(n,dtype=int)
    for k in range(len(probabilities_by_state)):
        mask=state_indices==k
        if mask.any():
            cum=np.cumsum(probabilities_by_state[k])
            result[mask]=np.minimum(np.searchsorted(cum,u[mask],side="right"),len(cum)-1)
    return result

def _inputs(cfg,market_ids):
    """For C8, edited C9 baseline/programme entries are *explicit overlays* on original C8 rows."""
    n=len(market_ids)
    profile=cfg["simulation"]["profile"]
    no_e=np.empty((n,3)); no_m=np.empty((n,3)); no_d=np.empty((n,3))
    changed={c:forecast_baseline(c,cfg) for c in NAMES}
    success={}
    for j,c in enumerate(NAMES):
        orig=BASE["no_action"][c]
        overlay={key:float(changed[c][key])-float(orig[key]) for key in ("ebitda","multiple","debt")}
        if profile=="C8 paired":
            for index,market in enumerate(STATES):
                hit=market_ids==index
                for matrix,field in ((no_e,"ebitda"),(no_m,"multiple"),(no_d,"debt")):
                    matrix[hit,j]=C8["no_action"][market][c][field]+overlay[field]
        else:
            no_e[:,j]=changed[c]["ebitda"]
            no_m[:,j]=changed[c]["multiple"]
            no_d[:,j]=changed[c]["debt"]
        no_m[:,j]=np.maximum(0,no_m[:,j])
        no_d[:,j]=np.maximum(0,no_d[:,j])
    for pname,p in cfg["programmes"].items():
        c=p["company"]; j=NAMES.index(c)
        eb=np.empty(n); mult=np.empty(n)
        if profile=="C8 paired":
            for i,market in enumerate(STATES):
                hit=market_ids==i
                eb[hit]=C8["success"][pname][market]["ebitda"]-C8["no_action"][market][c]["ebitda"]
                mult[hit]=C8["success"][pname][market]["multiple"]-C8["no_action"][market][c]["multiple"]
            eb+=p["funded_ebitda"]-BASE["programmes"][pname]["funded_ebitda"]
            mult+=p["funded_multiple"]-BASE["programmes"][pname]["funded_multiple"]
        else:
            eb[:]=p["funded_ebitda"]-BASE["no_action"][c]["ebitda"]
            mult[:]=p["funded_multiple"]-BASE["no_action"][c]["multiple"]
        success[pname]=(j,eb,mult,float(p.get("funded_debt_change",0)))
    return no_e,no_m,no_d,success

def run_simulation(cfg: dict):
    """Returns summary DataFrame, raw-by-strategy array dictionary and market/execution draws.

    Same market, EBITDA/multiple/debt shocks, milestone gates and closing uniforms across
    strategies. No-Action is computed *in each same draw*, not at a fixed Base value.
    """
    validate_simulation(cfg)
    s=cfg["simulation"]; n=int(s["n"])
    rng=np.random.default_rng(int(s["seed"]))
    if s["profile"]=="C8 paired":
        market_ids=rng.choice(3,n,p=validate_probabilities([s["market_probs"][x] for x in STATES]))
    else:
        market_ids=np.ones(n,dtype=int)
    base_e,base_m,base_d,success=_inputs(cfg,market_ids)
    rho=float(s["common_factor"])
    # Equicorrelation by a common-factor construction is PSD for all rho in [0,1].
    common=rng.standard_normal((n,3))
    idio=rng.standard_normal((n,3,3))
    z=np.sqrt(rho)*common[:,None,:]+np.sqrt(1-rho)*idio
    z=np.clip(z,-3,3)
    no_e=base_e*(1+z[:,:,0]*s["ebitda_vol"])
    no_m=np.maximum(0,base_m*(1+z[:,:,1]*s["multiple_vol"]))
    no_d=np.maximum(0,base_d+z[:,:,2]*s["debt_vol"])
    market_execution_prob=np.array([[s["execution_probs"][state][x] for x in EXECUTIONS] for state in STATES])
    shared_uniform=rng.random(n)
    execution_outcomes={}
    # Draw each programme once: same programme experiences same draw in every strategy.
    for pname in cfg["programmes"]:
        execution_outcomes[pname]=draw_categorical(
            rng, market_execution_prob,market_ids,n,
            shared_probability=s["execution_shared_probability"],common_uniform=shared_uniform)
    # Match transaction conditions at company level between no-action and every funded strategy.
    close_u_common=rng.random(n)
    closes=np.zeros((n,3),dtype=bool)
    for j,c in enumerate(NAMES):
        u=np.where(rng.random(n)<s["closing_shared_probability"],close_u_common,rng.random(n))
        closes[:,j]=(u<cfg["close_probabilities"][c]) & (cfg["exit_months"][c]<=24)
    ownership=np.array([cfg["ownership"][c] for c in NAMES],dtype=float)
    recovery=np.array([cfg["distressed_recovery"][c] for c in NAMES])
    contractual=np.array([cfg["contractual_receipts"][c] for c in NAMES])
    fee_factor=1-cfg["fees"]
    paid_fraction=1-cfg["escrow"]-cfg["deferred"]
    def value(earnings, multiples, debts):
        eq=exit_equity(earnings,multiples,debts,recovery[None,:])
        gross=eq*ownership[None,:]+contractual[None,:]
        return (gross*fee_factor*paid_fraction*closes).sum(axis=1)
    no_cash=value(no_e,no_m,no_d)+cfg["budget"]
    uniform_gates={pname:rng.random(n) for pname in cfg["programmes"]}
    outputs={}
    summaries=[]
    for strategy,info in eligible_strategies(cfg).items():
        eb=no_e.copy(); multiples=no_m.copy(); debt=no_d.copy()
        deployed=np.zeros(n)
        for pname in info["programmes"]:
            j,lift,premium,delta_debt=success[pname]
            outcomes=execution_outcomes[pname]
            fraction_e=np.array([C8["execution_states"][x]["ebitda_fraction"] for x in EXECUTIONS])[outcomes]
            fraction_m=np.array([C8["execution_states"][x]["multiple_fraction"] for x in EXECUTIONS])[outcomes]
            full_fund,_=funding_split(pname,cfg)
            if info["staged"]:
                upfront=cfg["milestone_upfront_fraction"]
                passed=uniform_gates[pname] < cfg["milestone_probability"]
                deployed+=full_fund*(upfront+(1-upfront)*passed)
                fraction_e=np.where(passed,fraction_e,0)
                fraction_m=np.where(passed,fraction_m,0)
            else:
                deployed+=full_fund
            eb[:,j]+=lift*fraction_e
            multiples[:,j]=np.maximum(0,multiples[:,j]+premium*fraction_m)
            debt[:,j]=np.maximum(0,debt[:,j]+delta_debt*fraction_e)
        cash=value(eb,multiples,debt)+cfg["budget"]-deployed
        increment=cash-no_cash
        tail_count=max(1,int(math.ceil(n*0.1)))
        # Lower-tail conditional average of paired incremental cash ("10% lower-tail CVaR").
        cvar=np.partition(increment,tail_count-1)[:tail_count].mean()
        row={
            "Strategy":strategy,"Priority":info["priority"],"Mean M24 cash":cash.mean(),
            "Median M24 cash":np.median(cash),
            "P5 M24 cash":np.percentile(cash,5),"P10 M24 cash":np.percentile(cash,10),
            "P90 M24 cash":np.percentile(cash,90),"P95 M24 cash":np.percentile(cash,95),
            "Mean incremental":increment.mean(),"Median incremental":np.median(increment),
            "P10 incremental":np.percentile(increment,10),
            "P(positive)":np.mean(increment>1e-9),"P(underperform)":np.mean(increment< -1e-9),
            "CVaR 10% incremental":cvar,
            "P(all exit by M24)":np.mean(np.all(closes,axis=1)),
            "Capital committed":info["deployed"],"Mean deployed":deployed.mean(),
            "P95 deployed":np.percentile(deployed,95)
        }
        summaries.append(row)
        outputs[strategy]={"cash":cash,"incremental":increment,"deployed":deployed,"closed":np.all(closes,axis=1)}
    return pd.DataFrame(summaries),outputs,{"market":market_ids,"execution":execution_outcomes,"seed":s["seed"],"profile":s["profile"]}

def raw_to_csv(summary,outputs,selected=None):
    import io
    names=list(outputs) if selected is None else selected
    rows=[]
    for name in names:
        d=outputs[name]
        rows.append(pd.DataFrame({"draw":np.arange(1,len(d["cash"])+1),
            "strategy":name, "terminal_cash":d["cash"],"incremental":d["incremental"],
            "capital_deployed":d["deployed"],"all_closed":d["closed"]}))
    return pd.concat(rows,ignore_index=True).to_csv(index=False)
