import io
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import statsmodels.api as sm

st.set_page_config(page_title="SA Rates RV Monitor", page_icon="📈", layout="wide")
st.title("South African Rates Relative Value Monitor")
st.caption("Spot ZARONIA and constant-maturity South African government bond relative-value analytics.")

@st.cache_data(ttl=3600)
def load_rates():
    url="https://rbond.co.za/api/v1/history.csv"
    r=requests.get(url,timeout=90)
    r.raise_for_status()
    raw=pd.read_csv(io.StringIO(r.text))
    raw["date"]=pd.to_datetime(raw["date"],errors="coerce")
    raw["yield_pct"]=pd.to_numeric(raw["yield_pct"],errors="coerce")
    wanted=["ZARONIA","ZARGB1","ZARGB5","ZARGB10"]
    raw=raw[raw["bond"].isin(wanted)].copy()
    raw=raw[raw["date"]>=pd.Timestamp("2023-09-25")]
    return (raw.pivot_table(index="date",columns="bond",values="yield_pct",aggfunc="last")
            .sort_index())

def build_spreads(rates):
    s=pd.DataFrame(index=rates.index)
    s["1Y - Spot ZARONIA"]=(rates["ZARGB1"]-rates["ZARONIA"])*100
    s["5Y - Spot ZARONIA"]=(rates["ZARGB5"]-rates["ZARONIA"])*100
    s["10Y - Spot ZARONIA"]=(rates["ZARGB10"]-rates["ZARONIA"])*100
    s["5Y - 1Y"]=(rates["ZARGB5"]-rates["ZARGB1"])*100
    s["10Y - 1Y"]=(rates["ZARGB10"]-rates["ZARGB1"])*100
    s["10Y - 5Y"]=(rates["ZARGB10"]-rates["ZARGB5"])*100
    return s

WINDOW=252
MIN_OBS=120
NEAR_UNIT_ROOT=0.995

def rolling_ar1_phi(series, window=WINDOW, min_obs=MIN_OBS):
    x=series.dropna().astype(float)
    out=pd.Series(np.nan,index=x.index,dtype=float)
    for i in range(len(x)):
        sample=x.iloc[max(0,i-window+1):i+1]
        if len(sample)<min_obs:
            continue
        reg=pd.DataFrame({"y":sample,"lag":sample.shift(1)}).dropna()
        X=sm.add_constant(reg["lag"])
        model=sm.OLS(reg["y"],X).fit()
        out.iloc[i]=float(model.params["lag"])
    return out

try:
    rates=load_rates()
except Exception as e:
    st.error("Unable to retrieve market data.")
    st.exception(e)
    st.stop()

spreads=build_spreads(rates)

rolling_phi=pd.DataFrame(index=spreads.index)
for col in spreads.columns:
    rolling_phi[col]=rolling_ar1_phi(spreads[col]).reindex(spreads.index)

rolling_mean=spreads.rolling(252,min_periods=60).mean()
rolling_std=spreads.rolling(252,min_periods=60).std()
rolling_z=(spreads-rolling_mean)/rolling_std

rows=[]
for name in spreads.columns:
    s=spreads[name].dropna()
    current=float(s.iloc[-1])
    as_of=s.index[-1]
    change_60=current-float(s.iloc[-61]) if len(s)>=61 else np.nan
    trailing=s.tail(252)
    z=(current-trailing.mean())/trailing.std(ddof=1) if len(trailing)>=60 and trailing.std(ddof=1)>0 else np.nan
    percentile=((s<=current).sum()/len(s))*100
    rp=rolling_phi[name].dropna()
    phi=float(rp.iloc[-1]) if not rp.empty else np.nan
    if 0<phi<NEAR_UNIT_ROOT:
        hl=-np.log(2)/np.log(phi)
        flag="Stationary"
    elif NEAR_UNIT_ROOT<=phi<1:
        hl=np.nan
        flag="Near unit root"
    elif phi>=1:
        hl=np.nan
        flag="Non-stationary AR(1)"
    else:
        hl=np.nan
        flag="Unavailable"
    rows.append({
        "Spread":name,"As Of":as_of,"Current (bp)":current,"60D Δ (bp)":change_60,
        "252D Z":z,"Historical %ile":percentile,"Rolling AR(1) φ":phi,
        "Half-Life (days)":hl,"Persistence":flag,"N":len(s)
    })

summary=pd.DataFrame(rows).set_index("Spread")
latest_source_date=rates.apply(lambda x:x.dropna().index.max()).max()
st.caption(f"Latest underlying market observation: {latest_source_date:%d %B %Y}")

st.subheader("Current Spreads")
top=["1Y - Spot ZARONIA","5Y - Spot ZARONIA","10Y - Spot ZARONIA"]
bot=["5Y - 1Y","10Y - 1Y","10Y - 5Y"]
for names in [top,bot]:
    cols=st.columns(3)
    for c,name in zip(cols,names):
        row=summary.loc[name]
        c.metric(name,f"{row['Current (bp)']:.1f} bp",f"{row['60D Δ (bp)']:+.1f} bp / 60D")

st.divider()
st.subheader("Relative-Value & Persistence")
display_tbl=summary[["As Of","Current (bp)","60D Δ (bp)","252D Z","Historical %ile","Rolling AR(1) φ","Half-Life (days)","Persistence"]].copy()
display_tbl["As Of"]=pd.to_datetime(display_tbl["As Of"]).dt.strftime("%Y-%m-%d")
st.dataframe(display_tbl.style.format({
    "Current (bp)":"{:.2f}","60D Δ (bp)":"{:+.2f}","252D Z":"{:+.2f}",
    "Historical %ile":"{:.1f}","Rolling AR(1) φ":"{:.4f}","Half-Life (days)":"{:.1f}"
}),use_container_width=True)

st.divider()
selected=st.selectbox("Select spread",spreads.columns.tolist())

series=spreads[selected].dropna()
fig=go.Figure(go.Scatter(x=series.index,y=series.values,mode="lines",name=selected))
fig.add_hline(y=0,line_dash="dash")
fig.update_layout(title=f"{selected} — Spread History",xaxis_title="Date",yaxis_title="Basis Points",hovermode="x unified",template="plotly_white",height=500)
st.plotly_chart(fig,use_container_width=True)

z=rolling_z[selected].dropna()
figz=go.Figure(go.Scatter(x=z.index,y=z.values,mode="lines",name="252D Z"))
for y,dash in [(0,"dash"),(2,"dot"),(-2,"dot")]:
    figz.add_hline(y=y,line_dash=dash)
figz.update_layout(title=f"{selected} — Rolling 252-Observation Z-Score",xaxis_title="Date",yaxis_title="Z-Score",hovermode="x unified",template="plotly_white",height=450)
st.plotly_chart(figz,use_container_width=True)

phi=rolling_phi[selected].dropna()
figp=go.Figure(go.Scatter(x=phi.index,y=phi.values,mode="lines",name="Rolling AR(1) φ"))
figp.add_hline(y=1.0,line_dash="dash",annotation_text="Unit root: φ = 1")
figp.add_hline(y=NEAR_UNIT_ROOT,line_dash="dot",annotation_text="Half-life threshold: φ = 0.995")
figp.update_layout(title=f"{selected} — Rolling 252-Observation AR(1) Persistence",xaxis_title="Date",yaxis_title="AR(1) φ",hovermode="x unified",template="plotly_white",height=500)
st.plotly_chart(figp,use_container_width=True)

with st.expander("Underlying data availability"):
    availability=pd.DataFrame({
        "First":rates.apply(lambda x:x.dropna().index.min()),
        "Last":rates.apply(lambda x:x.dropna().index.max()),
        "Observations":rates.count()
    })
    st.dataframe(availability,use_container_width=True)

with st.expander("Methodology"):
    st.markdown("""
- Analytics and calculations: author.
- Series: spot overnight ZARONIA plus 1Y, 5Y and 10Y constant-maturity South African government bond yields.
- The 1Y/5Y/10Y versus ZARONIA measures subtract the current spot overnight ZARONIA rate; they are not maturity-matched SAGB-versus-OIS spreads.
- All spreads are expressed in basis points.
- 252D Z-score uses the trailing 252 available observations, with a 60-observation minimum.
- Historical percentile is calculated over the full available sample from 25 September 2023.
- Rolling persistence uses an AR(1) fit over the latest 252 observations, with a 120-observation minimum.
- Half-life is shown only when 0 < φ < 0.995. Values at or above 0.995 are flagged instead of converted into unstable very-large half-lives.
- Missing observations are not forward-filled.
""")

st.divider()
st.caption("For research and informational purposes only.")
