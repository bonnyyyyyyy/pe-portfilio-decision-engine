"""Private Equity Portfolio Decision Engine | Streamlit entrypoint.

Launch from project root: streamlit run app.py
All forward inputs and execution probabilities are analyst-selected sensitivities.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from model import BASE, C8, COMPANIES, NAMES, default_config, compare, evaluate, validate_config, forecast_baseline
from strategies import STRATEGIES, eligible_strategies
from simulation import run_simulation, validate_simulation, raw_to_csv
from visualizations import (waterfall, comparison_chart, heatmap, tornado, outcomes_hist,
                             incremental_hist, ecdf_plot, downside_plot, risk_scatter,
                             beat_probability, pareto_plot)

st.set_page_config(page_title="PE | Portfolio Decision Engine",page_icon="📈",layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""<style>
:root {--pe-blue:#1e3557;--pe-teal:#168c8f;}
.block-container{padding-top:1.1rem;padding-bottom:2rem;max-width:1560px;}
h1,h2,h3{color:#1b3455;letter-spacing:-.025em;}
[data-testid="stMetric"]{border:1px solid #e2e9f0;border-radius:11px;
padding:15px 13px;background:#fbfcff;}
[data-testid="stMetricLabel"]{font-size:.88rem;}
[data-testid="stTabs"] button{font-weight:600;}
div[data-testid="stAlert"]{border-radius:8px;}
</style>""",unsafe_allow_html=True)

def hash_settings(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,default=str).encode()).hexdigest()

if "cfg" not in st.session_state:
    st.session_state.cfg=default_config()
if "run_cache" not in st.session_state:
    st.session_state.run_cache=None

def refresh_to(newcfg):
    st.session_state.clear()  # Also clears keyed widgets so reset/import truly takes effect.
    st.session_state.cfg=newcfg
    st.session_state.run_cache=None
    st.rerun()

cfg=st.session_state.cfg
with st.sidebar:
    st.title("Model controls")
    st.caption("USD millions · illustrative analyst forecasts")
    b1,b2=st.columns(2)
    if b1.button("Reset C9 Base",use_container_width=True):
        x=default_config()
        x["simulation"]["profile"]="C9 transaction base"
        refresh_to(x)
    if b2.button("Load C8 MC",use_container_width=True):
        x=default_config()
        x["simulation"]["profile"]="C8 paired"
        refresh_to(x)
    with st.expander("Fund budget & financing",expanded=True):
        cfg["budget"]=st.number_input("New capital available (official case: $15m)",min_value=0.0,max_value=100.0,
                                       value=float(cfg["budget"]),step=1.0)
        if abs(cfg["budget"]-15)>1e-8:
            st.warning("Counterfactual: official fund capital limit is $15m.")
        cfg["minimum_reserve"]=st.number_input("Protected minimum reserve ($m)",min_value=0.0,
                                                max_value=float(cfg["budget"]),value=min(float(cfg["minimum_reserve"]),float(cfg["budget"])),step=0.5)
        opts=["Fund only","Pro-rata co-funding","Confirmed co-funding"]
        cfg["financing_mode"]=st.selectbox("Financing structure",opts,index=opts.index(cfg["financing_mode"]) if cfg["financing_mode"] in opts else 0)
        if cfg["financing_mode"]!="Fund only":
            cfg["cofund_confirmed"]=st.checkbox("External shareholders explicitly agree to participate (illustrative)",
                                                 value=bool(cfg["cofund_confirmed"]))
            if cfg["financing_mode"]=="Pro-rata co-funding":
                st.caption("True pro-rata illustration: the fund pays its effective ownership percentage; other shareholders must actually participate.")
                for c in NAMES:
                    st.caption(f"{c}: fund {cfg['ownership'][c]:.0%} / co-shareholders {1-cfg['ownership'][c]:.0%}. For negotiated non-pro-rata sharing, use Confirmed co-funding.")
            else:
                st.caption("Each programme must be fully funded by the fund plus committed co-investors.")
        st.caption("Priced equity and shareholder loans are disabled: valuation, priority and repayment terms are unspecified. Exit-revenue values start at latest observed levels as placeholders, not independently supported exit projections.")

    with st.expander("Company forecasts & ownership",expanded=True):
        for c in NAMES:
            st.markdown(f"**{c}**")
            f=cfg["no_action"][c]
            st.caption(f"Historical: sales ${COMPANIES[c]['latest_revenue']:g}m · EBITDA ${COMPANIES[c]['latest_ebitda']:g}m")
            f["exit_revenue"]=st.number_input(f"{c} · exit revenue placeholder (latest observed by default)",min_value=0.0,
                                value=float(f["exit_revenue"]),step=10.0,key=f"rev_{c}")
            modes=["Direct EBITDA","Revenue × Margin"]
            cfg["forecast_modes"][c]=st.selectbox(f"{c} · EBITDA forecast method",modes,
                           index=modes.index(cfg["forecast_modes"][c]),key=f"mode_{c}")
            if cfg["forecast_modes"][c]=="Direct EBITDA":
                f["ebitda"]=st.number_input(f"{c} · No-Action exit EBITDA",value=float(f["ebitda"]),
                                            step=0.5,key=f"eb_{c}")
                implied=(f["ebitda"]/f["exit_revenue"]*100) if f["exit_revenue"] else 0
                st.caption(f"Implied forecast EBITDA margin: {implied:.2f}%")
            else:
                cfg["forecast_margins"][c]=st.number_input(f"{c} · forecast EBITDA margin (%)",
                    value=float(cfg["forecast_margins"][c]*100),step=0.5,
                    min_value=-100.0,max_value=100.0,key=f"margin_{c}")/100
                st.caption(f"Calculated EBITDA: ${f['exit_revenue']*cfg['forecast_margins'][c]:.2f}m")
            f["multiple"]=st.number_input(f"{c} · No-Action EV/EBITDA",value=float(f["multiple"]),
                    min_value=0.0,step=0.25,key=f"mult_{c}")
            f["debt"]=st.number_input(f"{c} · Exit net debt",value=float(f["debt"]),
                      min_value=0.0,step=2.0,key=f"debt_{c}")
            cfg["ownership"][c]=st.number_input(f"{c} · effective fund ownership (%)",
                    min_value=0.0,max_value=100.0,value=float(cfg["ownership"][c]*100),step=1.0,key=f"owner_{c}")/100
            cfg["distressed_recovery"][c]=st.number_input(f"{c} · explicit distressed equity recovery ($m)",
                min_value=0.0,value=float(cfg["distressed_recovery"][c]),step=1.0,key=f"recovery_{c}")
            st.divider()

    with st.expander("Funding programmes",expanded=False):
        for pname,p in cfg["programmes"].items():
            st.markdown(f"**{pname}** · {p['company']}")
            p["total_cost"]=st.number_input(f"{pname} · total company programme funding",min_value=0.0,
                              value=float(p["total_cost"]),step=1.0,key=f"cost_{pname}")
            if cfg["financing_mode"]=="Confirmed co-funding":
                cfg["fund_contributions"][pname]=st.number_input(f"{pname} · fund contribution",min_value=0.0,
                       max_value=float(p["total_cost"]),value=min(float(cfg["fund_contributions"][pname]),float(p["total_cost"])),step=0.5,key=f"fund_{pname}")
            orig=BASE["no_action"][p["company"]]
            p["funded_ebitda"]=orig["ebitda"]+st.number_input(
                f"{pname} · EBITDA improvement over original C9 No-Action ($m)",
                value=float(p["funded_ebitda"]-orig["ebitda"]),step=0.5,key=f"uplift_{pname}")
            p["funded_multiple"]=orig["multiple"]+st.number_input(
                f"{pname} · exit multiple premium vs original C9 (x)",
                value=float(p["funded_multiple"]-orig["multiple"]),
                min_value=-float(orig["multiple"]),step=0.25,key=f"premium_{pname}")
            p["funded_debt_change"]=st.number_input(f"{pname} · explicitly funded debt change ($m)",
                  value=float(p.get("funded_debt_change",0)),step=1.0,key=f"funddebt_{pname}")
            st.caption("C9 intervention delta follows the editable unfunded forecast; C8 retains its state-specific starting rows with explicitly documented overlays.")
            st.divider()
        cfg["milestone_upfront_fraction"]=st.slider("Staged variants · fraction spent upfront",0.0,1.0,
                                  float(cfg["milestone_upfront_fraction"]),0.05)
        cfg["milestone_probability"]=st.slider("Staged variants · milestone release probability (hypothetical)",0.0,1.0,
                                  float(cfg["milestone_probability"]),0.05)

    with st.expander("Exit settlement & transaction completion",expanded=False):
        cfg["fees"]=st.slider("Exit transaction fee rate (%)",0.0,20.0,float(cfg["fees"]*100),0.5)/100
        cfg["escrow"]=st.slider("Proceeds held in escrow beyond M24 (%)",0,50,int(round(cfg["escrow"]*100)),1)/100
        cfg["deferred"]=st.slider("Deferred past M24 (%)",0,50,int(round(cfg["deferred"]*100)),1)/100
        for c in NAMES:
            cfg["exit_months"][c]=st.number_input(f"{c} · closing month",min_value=1,max_value=48,
                        value=int(cfg["exit_months"][c]),key=f"month_{c}")
            cfg["close_probabilities"][c]=st.slider(f"{c} · P(close by M24), hypothetical",
                         0.0,1.0,float(cfg["close_probabilities"][c]),0.05,key=f"close_{c}")
            cfg["contractual_receipts"][c]=st.number_input(f"{c} · separately specified contractual cash ($m)",
                        min_value=0.0,value=float(cfg["contractual_receipts"][c]),step=1.0,key=f"contract_{c}")
        st.caption("Closing probabilities affect simulations; deterministic calculations use the explicit closing month. Escrow/deferred payments and non-closing NAV are NOT cash at M24.")

    with st.expander("Monte Carlo parameters",expanded=False):
        sim=cfg["simulation"]
        sim["profile"]=st.selectbox("Simulation profile",["C8 paired","C9 transaction base"],
                             index=["C8 paired","C9 transaction base"].index(sim["profile"]))
        sim["n"]=st.selectbox("Simulation draws",[1000,5000,10000,25000],
                      index=[1000,5000,10000,25000].index(sim["n"]) if sim["n"] in [1000,5000,10000,25000] else 2)
        sim["seed"]=st.number_input("Random seed",min_value=0,max_value=2**32-1,
                        value=int(sim["seed"]),step=1)
        st.markdown("**External market weights · hypothetical**")
        probs=sim["market_probs"]
        for k in ("Adverse","Base","Favorable"):
            probs[k]=st.number_input(f"P({k})",0.0,1.0,value=float(probs[k]),step=0.05,key=f"prob_{k}")
        st.markdown("**Execution conditional on market · hypothetical**")
        probability_table=pd.DataFrame.from_dict(sim["execution_probs"],orient="index")[["Success","Partial","Failure","Delay"]]
        edited=st.data_editor(probability_table,hide_index=False,use_container_width=True,
                              key="execution_editor",column_config={
                                state:st.column_config.NumberColumn(state,min_value=0,max_value=1,step=0.01,format="%.2f")
                                for state in ("Success","Partial","Failure","Delay")})
        sim["execution_probs"]=edited.to_dict(orient="index")
        sim["ebitda_vol"]=st.slider("Within-state EBITDA shock σ (relative)",0.0,0.6,float(sim["ebitda_vol"]),0.025)
        sim["multiple_vol"]=st.slider("Within-state multiple shock σ (relative)",0.0,0.6,float(sim["multiple_vol"]),0.025)
        sim["debt_vol"]=st.slider("Within-state exit debt shock σ (USD m)",0.0,30.0,float(sim["debt_vol"]),0.5)
        sim["common_factor"]=st.slider("Cross-company common-factor variance share",0.0,1.0,
                                  float(sim["common_factor"]),0.05)
        sim["execution_shared_probability"]=st.slider("Probability execution draw is shared",0.0,1.0,
                                  float(sim["execution_shared_probability"]),0.05)
        sim["closing_shared_probability"]=st.slider("Probability closing draw is shared",0.0,1.0,
                                  float(sim["closing_shared_probability"]),0.05)
        st.caption("Shock distributions: truncated normal at ±3σ, matched across strategies; common-factor model ensures valid covariance.")
    st.caption("Model 1.0 · C9 deterministic / C8 paired Monte Carlo · source: team PDF")

st.title("Portfolio Decision Engine")
st.caption("SKELAR × Horizon Capital × Mastercard Case Challenge 3.0  ·  24-month exit  ·  private equity investment committee")

st.warning("**Illustrative investment model — results are not verified fund forecasts.** Forward EBITDA, exit multiples, probabilities, funding terms and buy-side valuations are unverified analyst assumptions.",icon="⚠️")

try:
    validate_config(cfg)
    eligible=eligible_strategies(cfg)
except (ValueError,KeyError) as ex:
    st.error(f"Model inputs invalid: {ex}")
    st.stop()

if not eligible:
    st.error("No eligible strategies under the current fund budget and committed reserve.")
    st.stop()

heading_l,heading_r=st.columns([2,1])
with heading_l:
    selected=st.selectbox("Strategy under review",list(eligible),
       index=list(eligible).index("IT Full + Security Pilot") if "IT Full + Security Pilot" in eligible else 0)
with heading_r:
    secondary=st.selectbox("Side-by-side comparison",list(eligible),
                 index=0)
selected_result=compare(selected,cfg)
baseline=compare("No new deployment",cfg)

main_tabs=st.tabs(["Portfolio Overview & Sensitivity","Monte Carlo Simulation","Capital Allocation & Strategy Comparison"])

def money(v):
    return f"${v:,.2f}m"

with main_tabs[0]:
    st.subheader("M24 capital & realization")
    r=selected_result
    kpis=st.columns(6)
    metrics=[("Gross exit proceeds",money(r["gross"])),
             ("M24 total fund cash",money(r["terminal_cash"])),
             ("Capital deployed",money(r["deployed"])),
             ("Capital retained",money(r["retained"])),
             ("Incremental vs No-Action",f"${r['incremental']:+.2f}m"),
             ("Net Δ / $1 deployed",f"${r['return_per_dollar']:+.2f}" if r["return_per_dollar"] is not None else "N/A")]
    for col,(name,val) in zip(kpis,metrics):
        col.metric(name,val)
    st.caption("Net Δ / $1 deployed = (strategy M24 cash − matched No-Action M24 cash) ÷ actual new fund capital consumed; excludes historical fund cost.")
    if r["unrealized"]>0 or r["withheld"]>0:
        st.info(f"Unclosed exit NAV (excluded): {money(r['unrealized'])}. Escrow/deferred cash (excluded): {money(r['withheld'])}.")
    st.plotly_chart(waterfall(cfg,selected),use_container_width=True)
    st.subheader("Cash by company and transaction status")
    company_table=pd.DataFrame([
       {"Company":c,"Programme":p["programme"],"Exit EBITDA":p["ebitda"],"EV/EBITDA":p["multiple"],
        "Debt":p["debt"],"Equity value":p["equity"],"Gross fund receipts":p["gross"],
        "M24 realized fund cash":p["cash"],"Unrealized NAV":p["unrealized"],"Closed":p["closed"]}
       for c,p in r["companies"].items()])
    st.dataframe(company_table,use_container_width=True,hide_index=True)
    st.subheader("Live capital allocation comparison")
    fig_compare,comparison_data=comparison_chart(cfg)
    st.plotly_chart(fig_compare,use_container_width=True)
    st.subheader("Financial sensitivity heatmaps")
    c1,c2=st.columns([1,2])
    kind=c1.selectbox("Investment sensitivity",["IT Engineering","Security Hardware","E-commerce"])
    if kind=="IT Engineering":
        rangex=c1.slider("Exit EBITDA range",0.0,80.0,(12.0,35.0),1.0)
        rangey=c1.slider("Multiple range (x)",0.0,12.0,(3.0,8.0),0.25)
    elif kind=="Security Hardware":
        rangex=c1.slider("Premium range (x)",-2.0,3.0,(-0.5,1.5),0.1)
        rangey=c1.slider("Pilot cost range (USD m)",0.0,20.0,(0.0,10.0),0.5)
    else:
        rangex=c1.slider("Recovery EBITDA range",0.0,60.0,(0.0,35.0),1.0)
        rangey=c1.slider("Debt range (USD m)",0.0,100.0,(20.0,70.0),1.0)
    fig_heat,_=heatmap(cfg,kind,(*rangex,*rangey))
    c2.plotly_chart(fig_heat,use_container_width=True)
    st.caption("Heatmaps show one named programme against matched No-Action, including fund capital cost. The dark contour marks zero incremental fund cash.")
    st.subheader("Tornado: which assumptions move the decision?")
    t1,t2=st.columns(2)
    shock=t1.slider("Relative earnings/cost shock",0.05,0.50,0.20,0.05)
    multiple_shock=t2.slider("Absolute multiple shock",0.1,2.0,0.5,0.1)
    tornado_fig=tornado(cfg,selected,shock,multiple_shock)
    st.plotly_chart(tornado_fig,use_container_width=True)
    st.caption("A zero-length bar means that the selected strategy does not use that intervention under the specified cash and financing conditions. Debt shocks are in $10m steps.")
    st.download_button("Download live strategy comparison CSV",
                        comparison_data.to_csv(index=False).encode(),
                        file_name="c9_strategy_comparison.csv",mime="text/csv")
    if st.button("Prepare sensitivity chart PNG"):
        try:
            st.download_button("Download PNG",fig_heat.to_image(format="png",scale=2),
                               file_name="sensitivity.png",mime="image/png")
        except Exception as e:
            st.info("PNG export requires a compatible Plotly/Kaleido installation. Interactive HTML remains available.")

# Fingerprint excludes the currently selected strategy so simulations can be revisualized without rerunning.
fingerprint=hash_settings(cfg)
with main_tabs[1]:
    st.subheader("Paired Monte Carlo · risk/return analysis")
    st.markdown("One market state and the same continuous company shocks and M24 closing draw are used for **all strategies**, including the matched No-Action benchmark. Conditional execution draws depend on the shared market state.")
    sim_profile=cfg["simulation"]["profile"]
    st.caption(f"Active profile: {sim_profile}. C8 uses the PDF's distinct market-specific funded and No-Action assumptions. C9 uses transaction-date success premiums; execution probabilities remain illustrative. C9's later stressed table is not mixed into C8.")
    valid=True
    try:
        validate_simulation(cfg)
    except ValueError as ex:
        valid=False
        st.error(f"Cannot simulate until inputs are corrected: {ex}")
    if st.button("▶ Run Simulation",type="primary",disabled=not valid):
        with st.spinner("Running paired cash realizations..."):
            results, draws, metadata=run_simulation(copy.deepcopy(cfg))
            st.session_state.run_cache={"summary":results,"draws":draws,"metadata":metadata,
                                         "fingerprint":fingerprint,"date":datetime.now(timezone.utc).isoformat()}
    saved=st.session_state.run_cache
    if saved is None:
        st.info("Run the simulation to calculate paired risk distributions and portfolio rankings.")
    else:
        stale=saved["fingerprint"]!=fingerprint
        if stale:
            st.warning("⚠️ STALE SIMULATION: Financial or risk inputs have changed. The plots below reflect the **previous** simulation. Rerun before using these results in an investment decision or export.")
        else:
            st.success(f"{len(next(iter(saved['draws'].values()))['cash']):,} reproducible paired scenarios generated using seed {saved['metadata']['seed']}.")
        summary=saved["summary"];draws=saved["draws"]
        view=selected if selected in draws else "No new deployment"
        comparison_rows=summary[summary["Strategy"].isin([view,"No new deployment"])]
        cols=st.columns(4)
        row=summary.loc[summary["Strategy"]==view].iloc[0]
        for col,label,value in zip(cols,["Mean M24 cash","Mean paired incremental","P(positive paired Δ)","Worst 10% paired Δ"],
                                   [money(row["Mean M24 cash"]),money(row["Mean incremental"]),
                                    f"{row['P(positive)']:.1%}",money(row["CVaR 10% incremental"])]):
            col.metric(label,value)
        st.caption("All quantiles, probabilities and conditional expectations describe this synthetic parameter configuration only. CVaR is the mean paired incremental cash in the worst 10% of draws.")
        fig=outcomes_hist(view,draws)
        st.plotly_chart(fig,use_container_width=True)
        st.plotly_chart(incremental_hist(view,draws),use_container_width=True)
        st.subheader("Comparative tails & capital risk")
        names=st.multiselect("Strategies included in multi-strategy risk analysis",list(draws),
           default=list(dict.fromkeys([view,"No new deployment","IT Full","Security Pilot only"])))
        if names:
            st.plotly_chart(ecdf_plot(names,draws),use_container_width=True)
            st.plotly_chart(downside_plot(summary,names),use_container_width=True)
        c1,c2=st.columns(2)
        c1.plotly_chart(risk_scatter(summary),use_container_width=True)
        c2.plotly_chart(beat_probability(summary),use_container_width=True)
        st.subheader("Monte Carlo table · full conditional statistics")
        st.dataframe(summary.sort_values("Mean M24 cash",ascending=False),use_container_width=True,hide_index=True)
        if not stale:
            st.download_button("Download Monte Carlo summary CSV",summary.to_csv(index=False).encode(),
                              "monte_carlo_summary.csv","text/csv")
            if st.checkbox("Enable raw draw export (can be large)"):
                st.download_button("Download raw paired simulation CSV (selected strategies)",
                     raw_to_csv(summary,draws,names or [view]).encode(),"paired_draws.csv","text/csv")

with main_tabs[2]:
    st.subheader("Discrete capital allocation & investment constraints")
    st.markdown("Strategies are only ranked if funding is feasible and **no more than one intensive programme** is assigned. Partial and staged programmes retain their own explicit effectiveness assumptions.")
    eligible_rows=[]
    for name,meta in eligible.items():
        val=compare(name,cfg)
        eligible_rows.append({"Strategy":name,"Management priority":meta["priority"],
          "Fund deployed":val["deployed"],"Outside co-funding":meta["other_funding"],
          "Retained cash":val["retained"],"M24 fund cash":val["terminal_cash"],
          "Incremental cash":val["incremental"],"Δ per $ spent":val["return_per_dollar"]})
    comparison=pd.DataFrame(eligible_rows)
    metric=st.selectbox("Ranking objective",[
         "Max deterministic M24 cash","Max expected M24 cash",
         "Max expected incremental cash","Max probability of positive incremental",
         "Highest downside-protected cash subject to CVaR limit"])
    downside_limit=st.number_input("Minimum acceptable worst-10% paired incremental cash (USD m)",
                                   value=-12.0,step=1.0,disabled="downside" not in metric)
    cached=st.session_state.run_cache
    valid_run=cached is not None and cached["fingerprint"]==fingerprint
    if metric=="Max deterministic M24 cash":
        sorted_table=comparison.sort_values("M24 fund cash",ascending=False)
        rankkey="M24 fund cash"
    elif valid_run:
        s=cached["summary"]
        merged=comparison.merge(s,on="Strategy",how="inner")
        if metric=="Max expected M24 cash":
            rankkey="Mean M24 cash"
        elif metric=="Max expected incremental cash":
            rankkey="Mean incremental"
        elif metric=="Max probability of positive incremental":
            rankkey="P(positive)"
        else:
            # Select the highest expected cash among strategies passing a lower-tail cash hurdle.
            rankkey="Mean M24 cash"
            merged=merged[merged["CVaR 10% incremental"]>=downside_limit]
        sorted_table=merged.sort_values(rankkey,ascending=False)
    else:
        st.info("Run the Monte Carlo using the current assumptions to rank strategies by simulated objectives; showing deterministic ranking meanwhile.")
        sorted_table=comparison.sort_values("M24 fund cash",ascending=False)
        rankkey="M24 fund cash"
    if len(sorted_table):
        win=sorted_table.iloc[0]
        st.metric("Best eligible strategy under the SELECTED objective",win["Strategy"])
        st.caption(f"Objective: {metric} · ranking value: {win[rankkey]:.3f}; this is conditional on the current assumed financial, capital and risk parameters.")
    else:
        st.warning("No eligible strategy satisfies the chosen downside-protection limit.")
    st.dataframe(sorted_table,use_container_width=True,hide_index=True)
    if valid_run:
        st.plotly_chart(pareto_plot(cached["summary"]),use_container_width=True)
    st.subheader("Side-by-side investment strategy audit")
    left=compare(selected,cfg)
    right=compare(secondary,cfg)
    side=pd.DataFrame({"Measure":["M24 cash","Gross fund proceeds","Cash realized by M24",
           "Capital deployed","Other shareholders' funding","Retained fund cash",
           "Incremental cash","Escrow/deferred cash (excluded)","Unclosed NAV (excluded)"],
           f"{selected} (selected)":[left["terminal_cash"],left["gross"],left["realized"],left["deployed"],
                     left["other_shareholder_funding"],left["retained"],left["incremental"],
                     left["withheld"],left["unrealized"]],
           f"{secondary} (comparison)":[right["terminal_cash"],right["gross"],right["realized"],right["deployed"],
                      right["other_shareholder_funding"],right["retained"],right["incremental"],
                      right["withheld"],right["unrealized"]]})
    st.dataframe(side,use_container_width=True,hide_index=True)
    st.download_button("Download full live comparison CSV",comparison.to_csv(index=False).encode(),
                        "capital_allocation.csv","text/csv")
    st.markdown("**Constraints & dependencies.** The investment case assumes founder consent for material changes and sale of all three investments by M24; consent is not simulated as automatically received. If a staged milestone fails, released capital remains spent while unreleased capital stays with the fund; intervention effects are disabled unless the full required funding is supplied.")

with st.expander("Sources & Model Audit · provenance, equations and scenario configuration",expanded=False):
    st.markdown("""
**SOURCE HIERARCHY.** Official case constraints and the historical datapack are described in the team PDF pp. 7–12; C8 paired market operating assumptions and conditional execution are in PDF pp. 342–353, especially 346–348 and 352–353; C9 corrected $24m IT pricing EBITDA and cash reconciliation in pp. 371–376. Stage 11 pp. 66–68 are retained only for error-review context.

**C8 and C9 differ.** C8 success in Base assumes IT EBITDA **$25.5m** (M24) and a 5.5x exit multiple; C9 transaction-date Base instead uses **$24m**, yielding $152.400m instead of $154.875m for IT + Security Pilot. C8's conditional execution matrix and C9's later stress examples are *not* interchangeable.
""")
    st.code("""Enterprise value = max(0, exit EBITDA × exit multiple) when EBITDA > 0
Equity value = max(0, EV − exit net debt); otherwise separately entered distressed recovery
Gross fund proceeds = Equity value × effective fund ownership + explicitly entered contractual receipts
M24 cash proceeds = Gross × (1 − transaction fee rate) × (1 − escrow rate − deferred rate), if exit closes by M24
Terminal fund cash = Σ M24 cash proceeds + (initial fund capital − capital actually spent)
Paired incremental cash = strategy terminal cash − SAME-MARKET No-Action terminal cash
Return per $ new fund capital = paired incremental cash ÷ new fund capital actually spent
Execution fraction (Success/Partial/Failure/Delay): EBITDA = (1 / 0.5 / 0 / 0.6)
Execution fraction of multiple premium: (1 / 0.5 / 0 / 0.25)
""",language="text")
    st.markdown("**Data classification:** company historical facts are official illustrative datapack entries; forecast exit EBITDA, multiples, debt, pricing premiums and scenario probabilities are team analyst assumptions. Debt changes under funded programmes apply only when explicitly entered. No default dilution, loan recovery, preferred return, buy-side cash offer, deferred exit proceeds or shareholder participation has been invented.")
    st.markdown("**Simulation draws.** C8 uses the external state to select the corresponding No-Action and funded success rows; implementation probabilities depend on that same state. Funded EBITDA and multiple improvements are scaled by execution state and common market/debt shocks are paired. Editing C9 inputs overlays changes onto each C8 row relative to the source baseline, while keeping C8's discrete market and programme assumptions distinct.")
    st.json({"model_version":"1.0.0","configuration_timestamp_utc":datetime.now(timezone.utc).isoformat(),
             "profile":cfg["simulation"]["profile"],"seed":cfg["simulation"]["seed"],
             "execution_probability_model":"Conditional on shared market outcome",
             "notes":"Probabilities, correlations, volatilities, staged gates and programme effects are hypothetical sensitivities"})
    st.markdown("**User inputs (active configuration):**")
    st.dataframe(pd.DataFrame([{"company":c,**forecast_baseline(c,cfg),"ownership":cfg["ownership"][c],
                                "historical_latest_ebitda":COMPANIES[c]["latest_ebitda"]} for c in NAMES]),
                 hide_index=True,use_container_width=True)
    st.download_button("Export current scenario JSON",json.dumps(cfg,indent=2).encode(),
        "pe_scenario_assumptions.json","application/json")
    upload=st.file_uploader("Import a saved scenario configuration (JSON)",type=["json"])
    if upload is not None and st.button("Apply imported scenario"):
        try:
            loaded=json.loads(upload.getvalue().decode())
            validate_config(loaded)
            validate_simulation(loaded)
            refresh_to(loaded)
        except (ValueError,KeyError,TypeError,json.JSONDecodeError) as ex:
            st.error(f"Unable to load scenario: {ex}")
    st.caption("No database or external API is used. This app does not verify commercial financing terms, future valuations, buyer consent, or sale completion.")

st.caption("Financial model v1.0.0  ·  US$ millions  ·  Recalculated from the active assumptions  ·  Not investment advice")
