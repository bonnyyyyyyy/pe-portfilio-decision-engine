"""Interactive Plotly charts; all figures in USD millions."""
from __future__ import annotations
import copy
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from model import NAMES, BASE, compare, forecast_baseline, exit_equity, evaluate
from strategies import eligible_strategies, funding_split

LABELS={"E-commerce":"E-commerce","IT Engineering":"IT Engineering","Security Hardware":"Security Hardware"}

def theme(fig, height=440):
    fig.update_layout(template="plotly_white",height=height,font=dict(family="Arial, sans-serif"),
                      margin=dict(t=52,b=50,l=44,r=25),legend_title_text="",hovermode="closest")
    return fig

def waterfall(cfg,selected):
    cases=[evaluate("No new deployment",cfg),evaluate(selected,cfg)]
    fig=make_subplots(rows=1,cols=2,subplot_titles=["No new deployment",selected],
                      horizontal_spacing=0.12)
    for col,result in enumerate(cases,1):
        values=[result["companies"][c]["cash"] for c in NAMES]+[result["retained"]]
        fig.add_trace(go.Waterfall(x=NAMES+["Retained capital","M24 total"],
          y=values+[0],measure=["relative"]*4+["total"],name=result["strategy"],
          text=[f"${v:,.1f}m" for v in values]+[f"${result['terminal_cash']:,.1f}m"],
          textposition="outside",connector={"line":{"color":"#b1b9c8"}},
          increasing={"marker":{"color":"#197b85"}},totals={"marker":{"color":"#263b68"}},
          showlegend=False),row=1,col=col)
    fig.update_layout(title=f"Fund cash bridge | Δ vs No-Action: ${cases[1]['terminal_cash']-cases[0]['terminal_cash']:+,.2f}m",
        yaxis_title="USD m")
    return theme(fig,530)

def comparison_chart(cfg):
    rows=[]
    for name in eligible_strategies(cfg):
        r=compare(name,cfg)
        rows.append({"Strategy":name,"M24 cash":r["terminal_cash"],"Incremental":r["incremental"],"Deployed":r["deployed"]})
    data=pd.DataFrame(rows).sort_values("M24 cash")
    fig=px.bar(data,x="M24 cash",y="Strategy",orientation="h",color="Incremental",
               color_continuous_scale="RdYlGn",color_continuous_midpoint=0,
               hover_data={"Incremental":":.2f","Deployed":":.2f","M24 cash":":.2f"})
    fig.update_layout(yaxis_title="",xaxis_title="Terminal fund cash (USD m)",coloraxis_colorbar_title="Δ USD m")
    return theme(fig, max(430,27*len(data)+150)),data

def _unit_proceeds(company,ebitda,multiple,debt,cfg):
    eq=exit_equity(ebitda,multiple,debt,cfg["distressed_recovery"][company])
    multiple_cash=(1-cfg["fees"])*(1-cfg["escrow"]-cfg["deferred"])
    close=float(cfg["exit_months"][company] <=24)
    return (eq * cfg["ownership"][company]+cfg["contractual_receipts"][company])*multiple_cash*close

def heatmap(cfg,kind="IT Engineering",bounds=None,points=55):
    pmap={"IT Engineering":"IT Full","Security Hardware":"Security Pilot","E-commerce":"E-commerce Full"}
    pname=pmap[kind]; p=cfg["programmes"][pname]; base=forecast_baseline(kind,cfg)
    contribution,_=funding_split(pname,cfg)
    baseline=_unit_proceeds(kind,base["ebitda"],base["multiple"],base["debt"],cfg)
    if kind=="IT Engineering":
        bx=(max(0,base["ebitda"]-6),base["ebitda"]+19)
        by=(max(0,base["multiple"]-1),base["multiple"]+3)
        xlabel="Funded exit EBITDA (USD m)"; ylabel="Funded EV/EBITDA multiple"
    elif kind=="Security Hardware":
        bx=(-0.3,1.5); by=(0,max(7,contribution*1.8))
        xlabel="Pilot multiple premium (x)"; ylabel="Pilot fund cost (USD m)"
    else:
        bx=(0,max(35,p["funded_ebitda"]+10));by=(max(0,base["debt"]-25),base["debt"]+25)
        xlabel="Recovery exit EBITDA (USD m)"; ylabel="Funded exit net debt (USD m)"
    x0,x1,y0,y1 = bounds or (*bx,*by)
    x=np.linspace(x0,x1,points); y=np.linspace(y0,y1,points); xx,yy=np.meshgrid(x,y)
    if kind=="IT Engineering":
        z=_unit_proceeds(kind,xx,yy,base["debt"]+p.get("funded_debt_change",0),cfg)-baseline-contribution
    elif kind=="Security Hardware":
        z=_unit_proceeds(kind,base["ebitda"],base["multiple"]+xx,base["debt"],cfg)-baseline-yy
    else:
        z=_unit_proceeds(kind,xx,base["multiple"]+p["funded_multiple"]-BASE["no_action"][kind]["multiple"],yy,cfg)-baseline-contribution
    fig=go.Figure(go.Heatmap(x=x,y=y,z=z,colorscale="RdBu",zmid=0,
             colorbar=dict(title="Δ USD m"),hovertemplate=f"{xlabel}: %{{x:.2f}}<br>{ylabel}: %{{y:.2f}}<br>Incremental: $%{{z:.2f}}m<extra></extra>"))
    if z.min()<=0<=z.max():
        fig.add_trace(go.Contour(x=x,y=y,z=z,
             contours=dict(start=0,end=0,size=1,coloring="lines",showlabels=True),
             line=dict(color="#151c2c",width=2),showscale=False,hoverinfo="skip",name="Break-even"))
    fig.update_layout(title=f"{pname}: paired incremental fund cash",xaxis_title=xlabel,yaxis_title=ylabel)
    return theme(fig,470),(*bx,*by)

def tornado(cfg,selected,relative=0.2,absolute_multiple=0.5):
    baseline=compare(selected,cfg)["incremental"]
    tests=[]
    def copy_perturb(label,apply):
        val=[]
        for sign in (-1,1):
            temp=copy.deepcopy(cfg)
            apply(temp,sign)
            try:val.append(compare(selected,temp)["incremental"])
            except ValueError:val.append(float("nan"))
        tests.append((label,val[0]-baseline,val[1]-baseline))
    copy_perturb("IT funded EBITDA",lambda c,s: c["programmes"]["IT Full"].__setitem__("funded_ebitda",
                         max(-100, cfg["programmes"]["IT Full"]["funded_ebitda"]*(1+s*relative))))
    copy_perturb("IT funded multiple",lambda c,s: c["programmes"]["IT Full"].__setitem__("funded_multiple",
                         max(0,cfg["programmes"]["IT Full"]["funded_multiple"]+s*absolute_multiple)))
    copy_perturb("Security pilot premium",lambda c,s: c["programmes"]["Security Pilot"].__setitem__("funded_multiple",
                         max(0,cfg["programmes"]["Security Pilot"]["funded_multiple"]+s*absolute_multiple)))
    copy_perturb("Security pilot cost",lambda c,s: c["programmes"]["Security Pilot"].__setitem__("total_cost",
                         max(0,cfg["programmes"]["Security Pilot"]["total_cost"]*(1+s*relative))))
    copy_perturb("E-commerce funded EBITDA",lambda c,s: c["programmes"]["E-commerce Full"].__setitem__("funded_ebitda",
                         cfg["programmes"]["E-commerce Full"]["funded_ebitda"]*(1+s*relative)))
    copy_perturb("E-commerce closing net debt",lambda c,s: c["no_action"]["E-commerce"].__setitem__("debt",
                         max(0,cfg["no_action"]["E-commerce"]["debt"]+s*10)))
    if any(x < 1 for x in cfg["close_probabilities"].values()):
        copy_perturb("IT exit month (±2)",lambda c,s: c["exit_months"].__setitem__("IT Engineering",max(1,cfg["exit_months"]["IT Engineering"]+2*s)))
    tests=sorted(tests,key=lambda p:max(abs(p[1]),abs(p[2])),reverse=True)
    fig=go.Figure()
    fig.add_bar(x=[x[1] for x in tests],y=[x[0] for x in tests],orientation="h",name="Low input",marker_color="#b45d60")
    fig.add_bar(x=[x[2] for x in tests],y=[x[0] for x in tests],orientation="h",name="High input",marker_color="#29878b")
    fig.update_layout(barmode="overlay",title=f"Incremental cash sensitivity relative to selected Base (${baseline:+.2f}m)",
                      xaxis_title="Change in incremental cash (USD m)",yaxis=dict(autorange="reversed"))
    fig.add_vline(x=0,line_color="#182339")
    return theme(fig,370)

def outcomes_hist(selected,outputs):
    base=outputs["No new deployment"]["cash"]
    other=outputs[selected]["cash"]
    vals=pd.DataFrame({"Terminal cash":np.concatenate([base,other]),
         "Portfolio":["No-Action"]*len(base)+[selected]*len(other)})
    fig=px.histogram(vals,x="Terminal cash",color="Portfolio",barmode="overlay",
                     nbins=75,histnorm="probability density",opacity=0.64,
                     color_discrete_sequence=["#75849a","#1c858b"])
    fig.add_vline(x=other.mean(),line_dash="dash",line_color="#13777b",
                  annotation_text=f"Selected mean {other.mean():.1f}")
    fig.add_vline(x=np.percentile(other,10),line_dash="dot",line_color="#a15b4e",
                  annotation_text=f"Selected P10 {np.percentile(other,10):.1f}")
    fig.update_layout(xaxis_title="M24 terminal fund cash (USD m)",yaxis_title="Estimated density")
    return theme(fig,440)

def incremental_hist(selected,outputs):
    a=outputs[selected]["incremental"]
    data=pd.DataFrame({"Incremental M24 cash":a,"Outcome":np.where(a<0,"Negative","Nonnegative")})
    fig=px.histogram(data,x="Incremental M24 cash",color="Outcome",nbins=75,opacity=0.82,
                     color_discrete_map={"Negative":"#bc5f62","Nonnegative":"#218485"})
    fig.add_vline(x=0,line_width=2,line_color="#26354d")
    fig.add_vline(x=np.median(a),line_dash="dash",line_color="#bb9646")
    fig.update_layout(xaxis_title="Strategy - matched No-Action cash (USD m)",yaxis_title="Simulation draws")
    return theme(fig,410)

def ecdf_plot(names,outputs):
    fig=go.Figure()
    for name in names:
        a=np.sort(outputs[name]["incremental"])
        fig.add_trace(go.Scatter(x=a,y=np.arange(1,len(a)+1)/len(a),mode="lines",name=name))
    fig.add_vline(x=0,line_dash="dash",line_color="#26354d")
    fig.update_layout(title="Paired incremental cash CDF",xaxis_title="Incremental M24 cash (USD m)",
                      yaxis_title="Probability incremental cash ≤ x",yaxis_tickformat=".0%")
    return theme(fig,430)

def downside_plot(summary,selected_names):
    sub=summary[summary["Strategy"].isin(selected_names)].sort_values("Mean incremental")
    fig=go.Figure()
    fig.add_bar(y=sub["Strategy"],x=sub["Mean incremental"],orientation="h",name="Mean Δ",marker_color="#2b888b")
    fig.add_scatter(y=sub["Strategy"],x=sub["P10 incremental"],mode="markers",name="P10 Δ",marker=dict(symbol="diamond",size=11,color="#c9a15a"))
    fig.add_scatter(y=sub["Strategy"],x=sub["CVaR 10% incremental"],mode="markers",name="Worst 10% mean Δ",
                    marker=dict(symbol="x",size=11,color="#bd6765"))
    fig.add_vline(x=0,line_color="#1e2c40")
    fig.update_layout(title="Downside risk: mean, P10 and worst-decile shortfall",xaxis_title="Paired incremental fund cash (USD m)",yaxis_title="")
    return theme(fig,max(360,35*len(sub)+150))

def risk_scatter(summary):
    s=summary.copy()
    s["PointSize"]=np.maximum(1,s["Mean deployed"])
    fig=px.scatter(s,x="CVaR 10% incremental",y="Mean incremental",
        size="PointSize",color="Priority",hover_name="Strategy",
        hover_data={"Capital committed":":.2f","Mean deployed":":.2f","P(positive)":":.1%","CVaR 10% incremental":":.2f","PointSize":False},
        size_max=38)
    fig.add_hline(y=0,line_color="#aeb7bf",line_dash="dash")
    fig.update_layout(title="Risk / return by strategy",xaxis_title="Worst-decile mean incremental cash (USD m)",
                      yaxis_title="Mean paired incremental cash (USD m)")
    return theme(fig,470)

def beat_probability(summary):
    s=summary.sort_values("P(positive)")
    fig=px.bar(s,y="Strategy",x="P(positive)",orientation="h",color="P(positive)",color_continuous_scale="Teal")
    fig.update_layout(title="Probability of positive *paired* incremental fund cash",xaxis_title="Simulation share",
                      yaxis_title="",xaxis_tickformat=".0%",coloraxis_showscale=False)
    return theme(fig,max(460,27*len(s)+145))

def pareto_plot(summary):
    s=summary.sort_values("Mean deployed")
    is_pareto=[]
    for _,row in s.iterrows():
        dominated=any((s["Mean deployed"]<=row["Mean deployed"]) &
                      (s["Mean incremental"]>=row["Mean incremental"]) &
                      ((s["Mean deployed"]<row["Mean deployed"]) | (s["Mean incremental"]>row["Mean incremental"])))
        is_pareto.append(not dominated)
    s=s.copy(); s["Pareto"]=is_pareto
    fig=px.scatter(s,x="Mean deployed",y="Mean incremental",color="Pareto",
                   hover_name="Strategy",symbol="Pareto",
                   hover_data=["CVaR 10% incremental","P(positive)"])
    frontier=s[s["Pareto"]].sort_values("Mean deployed")
    fig.add_trace(go.Scatter(x=frontier["Mean deployed"],y=frontier["Mean incremental"],
                             mode="lines",line=dict(color="#25929c",dash="dot"),name="Non-dominated frontier",showlegend=True))
    fig.update_layout(title="Capital / expected incremental cash Pareto frontier",
                      xaxis_title="Mean new fund capital consumed (USD m)",
                      yaxis_title="Expected incremental M24 cash (USD m)")
    return theme(fig,450)
