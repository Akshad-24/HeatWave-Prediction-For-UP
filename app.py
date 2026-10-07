"""
Heatwave Dashboard - Foundations of Data Science
Run with:  streamlit run app.py
Needs FDS_DATASET.csv in the same folder as this file.
"""

import inspect
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score

st.set_page_config(page_title="Heatwave Dashboard", page_icon="🔥", layout="wide")

DATA_FILE = Path(__file__).parent / "FDS_DATASET.csv"
ALL = "All districts (average)"

# ---- Heatwave definition (IMD-style). Tweak these three numbers if you want a looser/stricter rule.
HOT_C = 40.0        # "hot day": max temperature >= 40 C
HW_ABS = 40.0       # heatwave day: max temp >= 40 C ...
HW_ANOM = 4.5       # ... AND at least 4.5 C above the normal for that district and date
HW_EXTREME = 45.0   # ... or max temp >= 45 C whatever the normal is
HORIZON = 3         # the model predicts: a heatwave day on any of the next 3 days
TEST_FROM_YEAR = 2023   # model learns from earlier years, is tested on 2023+

THERMAL = ["#1b1464", "#7b2d8e", "#d6336c", "#ff7a00", "#ffd60a"]   # thermal-camera scale, cold -> hot
RED, BLUE, ORANGE, PURPLE = "#d6336c", "#3a6ea5", "#ff7a00", "#7b2d8e"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

LABELS = {
    "T2M_MAX": "Max temperature (°C)", "T2M_MIN": "Min temperature (°C)", "T2M": "Mean temperature (°C)",
    "RH2M": "Relative humidity (%)", "QV2M": "Specific humidity (g/kg)", "TSOIL1": "Soil temperature (°C)",
    "ALLSKY_SFC_SW_DWN": "Solar radiation, actual (kWh/m²/day)",
    "CLRSKY_SFC_SW_DWN": "Solar radiation, clear sky (kWh/m²/day)", "CLOUD_AMT": "Cloud cover (%)",
    "GWETTOP": "Surface soil wetness (0-1)", "GWETROOT": "Root-zone soil wetness (0-1)",
    "WS10M": "Wind speed (m/s)", "WD10M": "Wind direction (°)", "PS": "Surface pressure (kPa)",
    "PRECTOTCORR": "Rainfall (mm/day)", "TS": "Land surface temperature (°C)", "EVLAND": "Land evaporation",
}
NUM_COLS = list(LABELS)

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;600;700;800&display=swap');

:root{
  --hw-radius:16px;
  --hw-border:rgba(128,128,128,.22);
  --hw-surface:rgba(128,128,128,.06);
  --hw-pad:1.1rem;
}

/* page frame */
.block-container{padding-top:2.4rem;padding-bottom:3.5rem;max-width:1280px}
header[data-testid="stHeader"]{background:transparent}
footer{visibility:hidden}

/* headings: Bricolage Grotesque everywhere a header appears */
h1,h2,h3,h4,
[data-testid="stHeading"] h1,[data-testid="stHeading"] h2,[data-testid="stHeading"] h3,
.hero-title,.kpi-value,.group-title,.result-big,.result-verdict{
  font-family:'Bricolage Grotesque',sans-serif !important;
  letter-spacing:-.015em;
}
h2,h3,[data-testid="stHeading"] h2,[data-testid="stHeading"] h3{font-weight:700 !important;margin-top:.4rem}

/* title block */
.hero-title{font-weight:800;font-size:2.8rem;line-height:1.05}
.hero-sub{opacity:.72;margin:.35rem 0 1rem;max-width:62ch}
.strip{display:flex;gap:4px;margin-bottom:1.4rem}
.cell{flex:1;padding:.85rem 0;text-align:center;color:#fff;text-shadow:0 1px 2px rgba(0,0,0,.5);
      line-height:1.3;border-radius:10px}
.cell span{display:block;font-size:.75rem}
.cell b{font-size:1.05rem}

/* soft containers, forms, widgets */
div[data-testid="stVerticalBlockBorderWrapper"]{border-radius:var(--hw-radius);border-color:var(--hw-border)}
div[data-baseweb="select"] > div,div[data-baseweb="input"],div[data-baseweb="base-input"]{border-radius:10px}
div[data-testid="stForm"]{border:none;padding:0;background:transparent}
div[data-testid="stExpander"]{border-radius:var(--hw-radius);border-color:var(--hw-border)}
div[data-testid="stAlert"]{border-radius:12px}
.stButton button,div[data-testid="stFormSubmitButton"] button{border-radius:12px;font-weight:600;padding:.65rem 1.2rem}

/* KPI cards */
.kpi{position:relative;overflow:hidden;height:100%;margin-bottom:.6rem;
     border:1px solid var(--hw-border);background:var(--hw-surface);
     border-radius:var(--hw-radius);padding:var(--hw-pad) var(--hw-pad) var(--hw-pad) 1.35rem}
.kpi::before{content:"";position:absolute;left:0;top:0;bottom:0;width:5px;background:var(--accent)}
.kpi-label{font-size:.88rem;font-weight:600;opacity:.75}
.kpi-value{font-weight:800;font-size:2.1rem;line-height:1.15;margin:.2rem 0 .15rem}
.kpi-sub{font-size:.8rem;opacity:.6}

/* native st.metric (used in the Predict tab) gets the same card look */
div[data-testid="stMetric"]{border:1px solid var(--hw-border);background:var(--hw-surface);
     border-left:4px solid #d6336c;border-radius:14px;padding:.85rem 1rem}

/* tabs */
div[data-baseweb="tab-list"]{gap:.4rem}
button[data-baseweb="tab"]{font-weight:600;padding:.6rem 1rem}

/* notes under the charts */
blockquote{border-left:4px solid #ff7a00 !important;background:var(--hw-surface);
           border-radius:0 12px 12px 0;padding:.8rem 1.1rem !important;margin:.8rem 0}
blockquote p{margin:0}

/* prediction form groups */
.group-title{font-weight:700;font-size:1.1rem;margin-bottom:.1rem}
.group-sub{font-size:.8rem;opacity:.6;margin-bottom:.7rem}

/* prediction result: the hero of the form */
.result{display:flex;flex-wrap:wrap;gap:2rem;align-items:center;margin:1.3rem 0 .6rem;
        padding:1.8rem 2rem;border-radius:22px;border:2px solid var(--rc);background:var(--rcbg)}
.result-left{min-width:170px}
.result-eyebrow{font-size:.88rem;font-weight:600;opacity:.75}
.result-big{font-weight:800;font-size:5rem;line-height:1;color:var(--rc)}
.result-right{flex:1;min-width:280px}
.result-verdict{font-weight:800;font-size:1.8rem;line-height:1.15}
.result-detail{opacity:.78;margin:.25rem 0 0}
.bar-wrap{position:relative;margin:1.1rem 0 1.9rem}
.bar{position:relative;height:16px;border-radius:99px;
     background:linear-gradient(90deg,#1b1464,#7b2d8e,#d6336c,#ff7a00,#ffd60a)}
.bar .cover{position:absolute;right:0;top:0;bottom:0;background:rgba(128,128,128,.3);border-radius:0 99px 99px 0}
.bar .thr{position:absolute;top:-6px;bottom:-6px;width:3px;margin-left:-1.5px;border-radius:2px;background:currentColor}
.thr-label{position:absolute;top:26px;transform:translateX(-50%);font-size:.78rem;opacity:.75;white-space:nowrap}
.chips{display:flex;flex-wrap:wrap;gap:.6rem}
.chip{border:1px solid var(--hw-border);border-radius:12px;padding:.45rem .8rem;background:rgba(128,128,128,.07)}
.chip span{display:block;font-size:.75rem;opacity:.65}
.chip b{font-size:.98rem}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def show(fig, height=400):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    kw = {"width": "stretch"} if "width" in inspect.signature(st.plotly_chart).parameters \
        else {"use_container_width": True}
    st.plotly_chart(fig, **kw)


def call_stretch(fn, *args, **kwargs):
    """Call a Streamlit widget at full width on both new (width=) and older (use_container_width=) versions."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except Exception:
        return fn(*args, use_container_width=True, **kwargs)


def note(text):
    st.markdown("> " + text)


def kpi_card(col, icon, label, value, sub, accent):
    col.markdown(
        f"<div class='kpi' style='--accent:{accent}'>"
        f"<div class='kpi-label'>{icon}&nbsp; {label}</div>"
        f"<div class='kpi-value'>{value}</div>"
        f"<div class='kpi-sub'>{sub}</div></div>", unsafe_allow_html=True)


def group_header(icon, title, sub):
    st.markdown(f"<div class='group-title'>{icon} {title}</div><div class='group-sub'>{sub}</div>",
                unsafe_allow_html=True)


def strength(r):
    a = abs(r)
    return ("very weak" if a < 0.2 else "weak" if a < 0.4 else "moderate" if a < 0.6
            else "strong" if a < 0.8 else "very strong")


def thermal(v, lo, hi):
    """Colour on the thermal scale for value v between lo and hi."""
    t = 0 if hi <= lo else min(max((v - lo) / (hi - lo), 0), 1) * (len(THERMAL) - 1)
    i = min(int(t), len(THERMAL) - 2)
    f = t - i
    a, b = [tuple(int(THERMAL[j][k:k + 2], 16) for k in (1, 3, 5)) for j in (i, i + 1)]
    return "rgb(%d,%d,%d)" % tuple(round(x + (y - x) * f) for x, y in zip(a, b))


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading dataset (first load only)...")
def load_data():
    df = pd.read_csv(DATA_FILE)
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
    df["city"] = df["city"].astype("category")
    for c in NUM_COLS + ["lat", "lon"]:
        df[c] = df[c].astype("float32")
    df = df.sort_values(["city", "date"]).reset_index(drop=True)
    df["year"] = df["date"].dt.year.astype("int16")
    df["month"] = df["date"].dt.month.astype("int8")
    df["doy"] = df["date"].dt.dayofyear.astype("int16")

    # "Normal" max temperature for every district and day of year (all years, smoothed over 15 days)
    clim = (df.groupby(["city", "doy"], observed=True)["T2M_MAX"].mean().unstack("doy")
              .reindex(columns=range(1, 367)).ffill(axis=1).bfill(axis=1))
    pad = pd.concat([clim.iloc[:, -7:], clim, clim.iloc[:, :7]], axis=1)
    clim = pad.T.rolling(15, center=True, min_periods=1).mean().T.iloc[:, 7:-7]
    grid = clim.reindex(df["city"].cat.categories).to_numpy()
    df["normal"] = grid[df["city"].cat.codes.to_numpy(), df["doy"].to_numpy() - 1].astype("float32")

    df["anom"] = (df["T2M_MAX"] - df["normal"]).astype("float32")      # degrees above normal
    df["hot"] = (df["T2M_MAX"] >= HOT_C).astype("float32")
    df["heat"] = (((df["T2M_MAX"] >= HW_ABS) & (df["anom"] >= HW_ANOM))
                  | (df["T2M_MAX"] >= HW_EXTREME)).astype("float32")   # heatwave day
    df["rainy"] = (df["PRECTOTCORR"] >= 1).astype("float32")
    return df


@st.cache_data(show_spinner=False)
def get_daily(city, y0, y1):
    """One row per date (a district, or the average of all districts)."""
    df = load_data()
    m = (df["year"] >= y0) & (df["year"] <= y1)
    if city != ALL:
        m &= df["city"] == city
    cols = NUM_COLS + ["hot", "heat", "anom", "rainy", "year", "month"]
    out = df.loc[m, ["date"] + cols].groupby("date").mean().reset_index()
    out[["year", "month"]] = out[["year", "month"]].round().astype(int)
    return out


@st.cache_data(show_spinner=False)
def get_districts(y0, y1):
    d = load_data()
    d = d[(d["year"] >= y0) & (d["year"] <= y1)]
    g = d.groupby("city", observed=True)[["T2M_MAX", "PRECTOTCORR", "hot", "heat", "lat", "lon"]].mean()
    out = pd.DataFrame({"Avg max temp (°C)": g["T2M_MAX"], "Annual rain (mm)": g["PRECTOTCORR"] * 365.25,
                        "Hot days / yr": g["hot"] * 365.25, "Heatwave days / yr": g["heat"] * 365.25,
                        "lat": g["lat"], "lon": g["lon"]})
    return out.reset_index().rename(columns={"city": "District"})


# ----------------------------------------------------------------------------
# Heatwave prediction model
# ----------------------------------------------------------------------------
FEATURES = ["T2M_MAX", "T2M_MIN", "anom", "tmax_3d", "tmax_change", "heat", "RH2M", "CLOUD_AMT",
            "PS", "WS10M", "GWETTOP", "PRECTOTCORR", "month"]
FEATURE_NAMES = {**LABELS, "anom": "Heat above normal (°C)", "tmax_3d": "3-day average max temp (°C)",
                 "tmax_change": "Max temp change vs 3 days ago (°C)", "heat": "Heatwave today",
                 "month": "Month of year"}

# Plain-language notes for the columns we derive ourselves (used by the "About the Data" tab)
DERIVED_NOTES = {
    "anom": "Max temperature minus the 15-day-smoothed normal for that district and day of year.",
    "tmax_3d": "Rolling mean of max temperature over the last 3 days, per district.",
    "tmax_change": "Max temperature today minus the value 3 days earlier, per district.",
    "heat": f"1 if max temp ≥ {HW_ABS:g} °C and ≥ {HW_ANOM:g} °C above normal, or ≥ {HW_EXTREME:g} °C. Otherwise 0.",
    "month": "Calendar month, 1 (Jan) to 12 (Dec).",
}


@st.cache_resource(show_spinner="Training the heatwave model (about 20-40 seconds, first time only)...")
def train_model():
    df = load_data()
    g = df.groupby("city", observed=True)
    d = df.assign(
        tmax_3d=g["T2M_MAX"].transform(lambda s: s.rolling(3, min_periods=1).mean()),
        tmax_change=g["T2M_MAX"].diff(3),
        # 1 if ANY of the next HORIZON days is a heatwave day
        target=pd.concat([g["heat"].shift(-k) for k in range(1, HORIZON + 1)], axis=1).max(axis=1, skipna=False),
    ).dropna(subset=["target"])
    train, test = d[d["year"] < TEST_FROM_YEAR], d[d["year"] >= TEST_FROM_YEAR]
    if train["target"].nunique() < 2 or test["target"].nunique() < 2:
        return None

    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=42)
    model.fit(train[FEATURES], train["target"])
    proba = model.predict_proba(test[FEATURES])[:, 1]
    y = test["target"].to_numpy().astype(int)

    sample = test.sample(min(20000, len(test)), random_state=1)
    imp = permutation_importance(model, sample[FEATURES], sample["target"], n_repeats=3,
                                 scoring="roc_auc", random_state=1)
    return {
        "model": model, "y": y, "proba": proba, "persist": test["heat"].to_numpy().astype(int),
        "auc": roc_auc_score(y, proba), "ap": average_precision_score(y, proba),
        "rate": float(y.mean()), "n_test": len(test),
        "importance": pd.Series(imp.importances_mean, index=FEATURES).sort_values(),
    }


# ----------------------------------------------------------------------------
# Data dictionary (built from LABELS and FEATURE_NAMES)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_dictionary():
    df = load_data()
    present = [c for c in FEATURE_NAMES if c in df.columns]
    agg = df[present].agg(["min", "mean", "max"]).T
    rows = []
    for code, label in FEATURE_NAMES.items():
        m = re.search(r"\s*\(([^)]*)\)\s*$", label)           # pull the unit out of "Name (unit)"
        raw = code in LABELS
        rows.append({
            "Column": code,
            "Description": label[:m.start()] if m else label,
            "Unit": m.group(1) if m else "-",
            "Type": "Raw measurement" if raw else "Derived feature",
            "In model": "✅" if code in FEATURES else "",
            "Min": float(agg.loc[code, "min"]) if code in agg.index else np.nan,
            "Mean": float(agg.loc[code, "mean"]) if code in agg.index else np.nan,
            "Max": float(agg.loc[code, "max"]) if code in agg.index else np.nan,
            "Notes": "Daily value as recorded in the dataset." if raw else DERIVED_NOTES.get(code, ""),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
if not DATA_FILE.exists():
    st.error(f"Could not find **{DATA_FILE.name}**. Put it in the same folder as app.py ({DATA_FILE.parent}).")
    st.stop()

data = load_data()
all_years = (int(data["year"].min()), int(data["year"].max()))
districts = sorted(data["city"].cat.categories.tolist())

st.markdown("<div class='hero-title'>Heatwave Monitor: Climate, Trends & Prediction</div>", unsafe_allow_html=True)

# ---- Global filters: one horizontal bar at the top
with st.container(border=True):
    f1, f2 = st.columns([1, 2], gap="large")
    city = f1.selectbox("District", [ALL] + districts, format_func=lambda c: c.replace("_", " "))
    y0, y1 = f2.slider("Years", all_years[0], all_years[1], all_years)

daily = get_daily(city, y0, y1)
if daily.empty:
    st.warning("No data for this selection.")
    st.stop()
n_years = max(y1 - y0 + 1, 1)

scope = "All districts (averaged)" if city == ALL else city.replace("_", " ")
monthly = daily.groupby("month").agg(tmax=("T2M_MAX", "mean"), tmin=("T2M_MIN", "mean"),
                                     rain=("PRECTOTCORR", "mean")).reindex(range(1, 13))
monthly["rain"] *= 30.44   # average mm per month

# ---- Month strip
lo, hi = float(monthly["tmax"].min()), float(monthly["tmax"].max())
cells = "".join(f"<div class='cell' style='background:{thermal(v, lo, hi)}'><span>{mo}</span>"
                f"<b>{v:.0f}°</b></div>" for mo, v in zip(MONTHS, monthly["tmax"]))
st.markdown(f"""<div class="hero-sub">{scope}, {y0} to {y1}. Each block is a month, coloured by its average daily max temperature.</div>
<div class="strip">{cells}</div>""", unsafe_allow_html=True)

# ---- KPI cards
hot_m, wet_m = int(monthly["tmax"].idxmax()), int(monthly["rain"].idxmax())
k = st.columns(4, gap="medium")
kpi_card(k[0], "🌡️", "Average max temperature", f"{daily['T2M_MAX'].mean():.1f} °C",
         f"Hottest month: {MONTHS[hot_m - 1]} ({monthly.loc[hot_m, 'tmax']:.0f} °C)", RED)
kpi_card(k[1], "🔥", "Heatwave days per year", f"{daily['heat'].sum() / n_years:.1f}",
         f"{daily['heat'].mean() * 100:.1f}% of all days", ORANGE)
kpi_card(k[2], "☀️", f"Days ≥ {HOT_C:g} °C per year", f"{daily['hot'].sum() / n_years:.0f}",
         f"{daily['hot'].mean() * 100:.0f}% of the year", PURPLE)
kpi_card(k[3], "🌧️", "Rain per year", f"{daily['PRECTOTCORR'].sum() / n_years:,.0f} mm",
         f"Wettest month: {MONTHS[wet_m - 1]}", BLUE)

with st.expander("ℹ️ How is a heatwave day defined?"):
    st.markdown(f"A **heatwave day** has a max temperature of at least **{HW_ABS:g} °C** and at least "
                f"**{HW_ANOM:g} °C above the normal** for that district and date, or a max temperature of "
                f"**{HW_EXTREME:g} °C or more**. The Districts and Predict tabs always use all districts.")


# ----------------------------------------------------------------------------
# Predict tab pieces
# ----------------------------------------------------------------------------
def result_hero(p, thr, chips):
    warn = p >= thr
    color, bg = ("#d6336c", "rgba(214,51,108,.12)") if warn else ("#2b9348", "rgba(43,147,72,.12)")
    verdict = "🚨 Heatwave warning" if warn else "✅ No warning"
    detail = (f"{p * 100:.0f}% is above your {thr * 100:.0f}% threshold." if warn
              else f"{p * 100:.0f}% is below your {thr * 100:.0f}% threshold.")
    chip_html = "".join(f"<div class='chip'><span>{a}</span><b>{b}</b></div>" for a, b in chips)
    label_at = min(max(thr, 0.12), 0.88) * 100
    st.markdown(
        f"<div class='result' style='--rc:{color};--rcbg:{bg}'>"
        f"<div class='result-left'><div class='result-eyebrow'>Heatwave risk in the next {HORIZON} days</div>"
        f"<div class='result-big'>{p * 100:.0f}%</div></div>"
        f"<div class='result-right'><div class='result-verdict'>{verdict}</div>"
        f"<div class='result-detail'>{detail}</div>"
        f"<div class='bar-wrap'><div class='bar'><div class='cover' style='width:{(1 - p) * 100:.1f}%'></div>"
        f"<div class='thr' style='left:{thr * 100:.1f}%'></div></div>"
        f"<div class='thr-label' style='left:{label_at:.1f}%'>Your threshold: {thr * 100:.0f}%</div></div>"
        f"<div class='chips'>{chip_html}</div></div></div>", unsafe_allow_html=True)


def result_placeholder():
    st.markdown(
        "<div class='result' style='--rc:#9aa0a6;--rcbg:rgba(128,128,128,.06)'>"
        "<div class='result-right'><div class='result-verdict'>🔮 Your result will appear here</div>"
        "<div class='result-detail'>Set the conditions above, then press <b>Check heatwave risk</b>.</div>"
        "</div></div>", unsafe_allow_html=True)


def predict_tab():
    st.subheader(f"Will a heatwave hit in the next {HORIZON} days?")
    st.write(f"A gradient-boosting model reads **today's weather and the recent build-up of heat** and estimates "
             f"whether any of the **next {HORIZON} days** will be a heatwave day. It learns from every district "
             f"using data before {TEST_FROM_YEAR} and is tested on {TEST_FROM_YEAR} onwards, which it has never "
             f"seen. It ignores the district and year filters.")
    M = train_model()
    if M is None:
        st.warning("There are not enough heatwave days before/after the split year to train a model. "
                   "Lower HW_ANOM or HW_ABS, or change TEST_FROM_YEAR, at the top of app.py.")
        return

    thr = st.slider("Warning threshold (how sure the model must be before it warns)", 0.05, 0.90, 0.30, 0.05)
    pred, y, p0 = (M["proba"] >= thr).astype(int), M["y"], M["persist"]
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    fn, tn = int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    ptp = int(((p0 == 1) & (y == 1)).sum())
    prec_p, rec_p = ptp / max(int(p0.sum()), 1), ptp / max(int(y.sum()), 1)

    m = st.columns(5)
    m[0].metric("Heatwaves caught (recall)", f"{rec * 100:.0f} %")
    m[1].metric("Warnings that were right", f"{prec * 100:.0f} %")
    m[2].metric("ROC AUC", f"{M['auc']:.2f}")
    m[3].metric("'Heatwave today' rule: recall", f"{rec_p * 100:.0f} %")
    m[4].metric("'Heatwave today' rule: precision", f"{prec_p * 100:.0f} %")

    c1, c2 = st.columns(2)
    with c1:
        cm = pd.DataFrame([[tn, fp], [fn, tp]], index=["Actual: no heatwave", "Actual: heatwave"],
                          columns=["Warned: no", "Warned: yes"])
        fig = px.imshow(cm, text_auto=",d", color_continuous_scale=THERMAL, title="Confusion matrix (test data)")
        fig.update_layout(coloraxis_showscale=False)
        show(fig, 370)
    with c2:
        imp = M["importance"].rename(index=FEATURE_NAMES)
        fig = px.bar(x=imp.values, y=imp.index, orientation="h", color_discrete_sequence=[RED],
                     labels={"x": "Drop in AUC when shuffled", "y": ""}, title="Which inputs matter most?")
        show(fig, 370)
    top = imp.sort_values(ascending=False).index[:3].tolist()
    note(f"Heatwaves are rare: only **{M['rate'] * 100:.1f}%** of the {M['n_test']:,} test days have one coming in "
         f"the next {HORIZON} days, so a model that always says 'no heatwave' would be right "
         f"{100 - M['rate'] * 100:.1f}% of the time. That is why accuracy is not shown; recall and precision tell "
         f"the real story. At this threshold the model catches **{rec * 100:.0f}%** of upcoming heatwave days and "
         f"**{prec * 100:.0f}%** of its warnings are correct, versus {rec_p * 100:.0f}% and {prec_p * 100:.0f}% "
         f"for the simple rule 'heatwave today means heatwave soon'. Lowering the threshold catches more "
         f"heatwaves but raises false alarms. ROC AUC is {M['auc']:.2f} and PR-AUC is {M['ap']:.2f} (random "
         f"guessing would score {M['rate']:.2f}). Most useful inputs: **{top[0]}**, **{top[1]}**, **{top[2]}**.")

    # ------------------------------------------------------------ Try it yourself
    st.subheader("Try it yourself")
    st.caption("Describe a day's weather. Sliders are for values that move a lot day to day, "
               "number boxes are for precise readings.")
    with st.form("predict"):
        g = st.columns(4, gap="medium")

        with g[0], st.container(border=True):
            group_header("🌡️", "Temperature data", "Today's readings and the seasonal normal")
            tmax = st.slider("Max temp today (°C)", 5.0, 52.0, 41.0, 0.5)
            tmin = st.slider("Min temp today (°C)", 0.0, 40.0, 26.0, 0.5)
            normal = st.number_input("Normal max for this date (°C)", 5.0, 50.0, 37.0, 0.5)

        with g[1], st.container(border=True):
            group_header("📈", "Recent heat build-up", "How the last few days compare")
            month = st.selectbox("Month", range(1, 13), index=3, format_func=lambda k: MONTHS[k - 1])
            avg3 = st.number_input("Avg max, last 3 days (°C)", 5.0, 52.0, 40.0, 0.5)
            ago3 = st.number_input("Max temp 3 days ago (°C)", 5.0, 52.0, 38.0, 0.5)

        with g[2], st.container(border=True):
            group_header("☁️", "Atmospheric conditions", "Sky and air movement")
            cloud = st.slider("Cloud cover (%)", 0.0, 100.0, 15.0, 1.0)
            ps = st.number_input("Surface pressure (kPa)", 90.0, 103.0, 98.5, 0.1)
            ws = st.number_input("Wind speed (m/s)", 0.0, 15.0, 3.0, 0.1)

        with g[3], st.container(border=True):
            group_header("💧", "Moisture and rain", "How dry the air and ground are")
            rh = st.slider("Relative humidity (%)", 0.0, 100.0, 25.0, 1.0)
            soil = st.slider("Soil wetness (0 dry - 1 wet)", 0.0, 1.0, 0.2, 0.01)
            rain = st.number_input("Rainfall today (mm)", 0.0, 350.0, 0.0, 0.5)

        go_btn = call_stretch(st.form_submit_button, "🔥 Check heatwave risk", type="primary")

    if go_btn:
        anom = tmax - normal
        heat_now = float((tmax >= HW_ABS and anom >= HW_ANOM) or tmax >= HW_EXTREME)
        row = pd.DataFrame([{"T2M_MAX": tmax, "T2M_MIN": tmin, "anom": anom, "tmax_3d": avg3,
                             "tmax_change": tmax - ago3, "heat": heat_now, "RH2M": rh, "CLOUD_AMT": cloud,
                             "PS": ps, "WS10M": ws, "GWETTOP": soil, "PRECTOTCORR": rain, "month": month}])[FEATURES]
        p = float(M["model"].predict_proba(row)[0, 1])
        # Remember the last result so moving the threshold slider updates the verdict instead of clearing it
        st.session_state["hw_result"] = {"p": p, "anom": anom, "chg": tmax - ago3, "heat_now": heat_now}

    res = st.session_state.get("hw_result")
    if res:
        result_hero(res["p"], thr, [
            ("Heat above normal", f"{res['anom']:+.1f} °C"),
            ("Change vs 3 days ago", f"{res['chg']:+.1f} °C"),
            ("Counts as a heatwave day today", "Yes" if res["heat_now"] else "No"),
        ])
        st.caption("Today counts as a heatwave day in this input if it meets the definition at the top of the page. "
                   "Unusual combinations may give odd answers, because the model has mostly seen realistic weather.")
    else:
        result_placeholder()


# ----------------------------------------------------------------------------
# About the data tab
# ----------------------------------------------------------------------------
def about_tab():
    d0, d1 = data["date"].min(), data["date"].max()
    n_rows = len(data)
    per_dist = n_rows / max(len(districts), 1)

    st.subheader("What is in the dataset?")
    s = st.columns(4, gap="medium")
    kpi_card(s[0], "🧾", "Daily records", f"{n_rows:,}", "One row is one district on one day", RED)
    kpi_card(s[1], "📅", "Date range", f"{all_years[0]} to {all_years[1]}",
             f"{d0:%d %b %Y} to {d1:%d %b %Y}", ORANGE)
    kpi_card(s[2], "📍", "Districts", f"{len(districts)}", f"About {per_dist:,.0f} days each", PURPLE)
    kpi_card(s[3], "🧪", "Weather measurements", f"{len(NUM_COLS)}", "Plus latitude and longitude", BLUE)

    note(f"The dataset holds **{n_rows:,} daily records** for **{len(districts)} districts** between "
         f"**{d0:%d %b %Y}** and **{d1:%d %b %Y}** (about {per_dist:,.0f} days per district). Each row has "
         f"{len(NUM_COLS)} weather measurements. From these we derive a 'normal' temperature for every district "
         f"and date, the heat above normal, and a 0/1 heatwave flag. The model learns from "
         f"{all_years[0]} to {TEST_FROM_YEAR - 1} and is tested on {TEST_FROM_YEAR} onwards.")

    e1, e2 = st.columns(2, gap="medium")
    with e1:
        with st.expander("📍 Districts covered"):
            st.write(", ".join(d.replace("_", " ") for d in districts))
    with e2:
        with st.expander("🔍 First rows of the data"):
            call_stretch(st.dataframe, data[["date", "city"] + NUM_COLS].head(10), hide_index=True)

    st.subheader("Data dictionary")
    view = st.radio("Show", ["All columns", "Raw measurements", "Derived features", "Used by the model"],
                    horizontal=True, label_visibility="collapsed")
    dd = build_dictionary()
    if view == "Raw measurements":
        dd = dd[dd["Type"] == "Raw measurement"]
    elif view == "Derived features":
        dd = dd[dd["Type"] == "Derived feature"]
    elif view == "Used by the model":
        dd = dd[dd["In model"] == "✅"]
    call_stretch(st.dataframe, dd, hide_index=True, height=min(80 + 35 * len(dd), 800), column_config={
        "Column": st.column_config.TextColumn("Column", help="Column name in the dataset or model"),
        "In model": st.column_config.TextColumn("In model", help="Used as an input by the prediction model"),
        "Min": st.column_config.NumberColumn("Min", format="%.2f"),
        "Mean": st.column_config.NumberColumn("Mean", format="%.2f"),
        "Max": st.column_config.NumberColumn("Max", format="%.2f"),
        "Notes": st.column_config.TextColumn("Notes", width="large"),
    })
    st.caption("Min, mean and max cover all districts and all years. "
               "The 3-day average and 3-day change are calculated while training the model, so they have no range here.")


# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
t_clim, t_heat, t_link, t_dist, t_pred, t_about = st.tabs(
    ["🌡️ Climate", "🔥 Heatwaves", "🔗 Links", "🗺️ Districts", "🤖 Predict", "📖 About the Data"])

# ---------------------------------------------------------------- Climate
with t_clim:
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=MONTHS, y=monthly["tmax"], name="Max", line=dict(color=RED, width=3)))
        fig.add_trace(go.Scatter(x=MONTHS, y=monthly["tmin"], name="Min", line=dict(color=BLUE, width=3),
                                 fill="tonexty", fillcolor="rgba(255,122,0,.12)"))
        fig.update_layout(title="Average temperature by month (°C)", legend=dict(orientation="h", y=-0.15))
        show(fig, 360)
    with c2:
        fig = px.bar(x=MONTHS, y=monthly["rain"], color_discrete_sequence=[BLUE],
                     labels={"x": "", "y": "mm per month"}, title="Average rainfall by month")
        show(fig, 360)
    hot_m, cold_m, wet_m = int(monthly["tmax"].idxmax()), int(monthly["tmax"].idxmin()), int(monthly["rain"].idxmax())
    share = monthly.loc[[6, 7, 8, 9], "rain"].sum() / monthly["rain"].sum() * 100
    note(f"Max temperature peaks in **{MONTHS[hot_m - 1]}** ({monthly.loc[hot_m, 'tmax']:.1f} °C) and is lowest "
         f"in **{MONTHS[cold_m - 1]}** ({monthly.loc[cold_m, 'tmax']:.1f} °C), a swing of "
         f"{monthly.loc[hot_m, 'tmax'] - monthly.loc[cold_m, 'tmax']:.0f} °C. **{MONTHS[wet_m - 1]}** is the "
         f"wettest month (about {monthly.loc[wet_m, 'rain']:.0f} mm) and June to September bring "
         f"**{share:.0f}%** of the yearly rain. Heatwave risk sits in the hot, dry months before the rains.")

# -------------------------------------------------------------- Heatwaves
with t_heat:
    if city == ALL:
        st.caption("All-districts view: counts are the average number of heatwave days per district.")
    yearly = daily.groupby("year").agg(heat=("heat", "sum"), hot=("hot", "sum")).reset_index()
    trend = ""
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(yearly, x="year", y="heat", color="heat", color_continuous_scale=THERMAL,
                     labels={"heat": "Heatwave days", "year": "Year"}, title="Heatwave days per year")
        fig.update_layout(coloraxis_showscale=False)
        if len(yearly) >= 4:
            res = stats.linregress(yearly["year"], yearly["heat"])
            fig.add_trace(go.Scatter(x=yearly["year"], y=res.intercept + res.slope * yearly["year"],
                                     name="Trend", line=dict(color="#9aa0a6", dash="dash")))
            sig = "statistically significant" if res.pvalue < 0.05 else "not statistically significant"
            trend = (f"The trend is **{res.slope * 10:+.1f} heatwave days per decade** "
                     f"(p = {res.pvalue:.2f}, {sig}). ")
        show(fig, 380)
    with c2:
        pv = (daily.pivot_table(index="year", columns="month", values="heat", aggfunc="sum")
                   .reindex(columns=range(1, 13)).fillna(0))
        pv.columns = MONTHS
        fig = px.imshow(pv, aspect="auto", color_continuous_scale=THERMAL,
                        labels=dict(x="Month", y="Year", color="Days"), title="Heatwave days: year × month")
        show(fig, 380)
    by_m = daily.groupby("month")["heat"].sum() / n_years
    if yearly["heat"].sum() == 0:
        note(f"There are no heatwave days under the current definition in this selection. Try another "
             f"district, or loosen HW_ANOM / HW_ABS at the top of app.py.")
    else:
        peak_y, peak_m = int(yearly.loc[yearly["heat"].idxmax(), "year"]), int(by_m.idxmax())
        note(f"There are on average **{yearly['heat'].mean():.1f} heatwave days per year** "
             f"({daily['heat'].mean() * 100:.1f}% of days). The worst year was **{peak_y}** with "
             f"{yearly['heat'].max():.0f} days, and heatwaves cluster in **{MONTHS[peak_m - 1]}** "
             f"({by_m[peak_m]:.1f} days per year on average). {trend}The year × month map shows whether "
             f"heatwaves are a steady yearly feature or come in a few bad years.")

# ------------------------------------------------------------------ Links
with t_link:
    corr_cols = ["T2M_MAX", "T2M_MIN", "RH2M", "CLOUD_AMT", "PS", "WS10M", "GWETTOP",
                 "ALLSKY_SFC_SW_DWN", "PRECTOTCORR", "anom", "heat"]
    short = {"T2M_MAX": "Max temp", "T2M_MIN": "Min temp", "RH2M": "Humidity", "CLOUD_AMT": "Cloud",
             "PS": "Pressure", "WS10M": "Wind", "GWETTOP": "Soil wetness", "ALLSKY_SFC_SW_DWN": "Sunlight",
             "PRECTOTCORR": "Rain", "anom": "Above normal", "heat": "Heatwave"}
    corr = daily[corr_cols].corr()
    fig = px.imshow(corr.rename(index=short, columns=short), text_auto=".2f", zmin=-1, zmax=1,
                    color_continuous_scale="RdBu_r", title="What moves with heatwaves? (daily values)")
    show(fig, 540)
    hc = corr["heat"].drop(["heat", "T2M_MAX", "anom"]).dropna()
    if len(hc) >= 2:
        t1, t2 = hc.abs().sort_values(ascending=False).index[:2]
        note(f"Apart from temperature itself, heatwave days are most linked to **{short[t1]}** "
             f"(r = {hc[t1]:.2f}) and **{short[t2]}** (r = {hc[t2]:.2f}). Dry air, clear skies and "
             f"dry soil usually go together with extreme heat. These values mix all seasons, so correlation "
             f"does not prove one thing causes the other.")

    st.subheader("Explore any two variables")
    c1, c2 = st.columns(2)
    x_var = c1.selectbox("X axis", NUM_COLS, index=NUM_COLS.index("RH2M"), format_func=lambda k: LABELS[k])
    y_var = c2.selectbox("Y axis", NUM_COLS, index=NUM_COLS.index("T2M_MAX"), format_func=lambda k: LABELS[k])
    if x_var == y_var:
        st.warning("Choose two different variables.")
    else:
        sm = daily.sample(min(len(daily), 3000), random_state=0).copy()
        sm["Day type"] = np.where(sm["heat"] > 0, "Heatwave day", "Normal day")
        fig = px.scatter(sm, x=x_var, y=y_var, color="Day type", opacity=0.5,
                         color_discrete_map={"Heatwave day": RED, "Normal day": BLUE},
                         labels={x_var: LABELS[x_var], y_var: LABELS[y_var]},
                         title=f"{LABELS[y_var]} vs {LABELS[x_var]} (sample of {len(sm):,} days)")
        res = stats.linregress(daily[x_var], daily[y_var])
        xs = np.linspace(daily[x_var].min(), daily[x_var].max(), 50)
        fig.add_trace(go.Scatter(x=xs, y=res.intercept + res.slope * xs, mode="lines", name="Best fit",
                                 line=dict(color="#9aa0a6", width=3)))
        show(fig, 460)
        r = res.rvalue
        note(f"There is a **{strength(r)} {'positive' if r > 0 else 'negative'}** relationship "
             f"(r = {r:.2f}, R² = {r ** 2:.2f}). When {LABELS[x_var].lower()} rises by one unit, "
             f"{LABELS[y_var].lower()} changes by about **{res.slope:+.3f}** on average. The line explains "
             f"{r ** 2 * 100:.0f}% of the day-to-day variation; the rest comes from other factors.")

# -------------------------------------------------------------- Districts
with t_dist:
    dist = get_districts(y0, y1)
    metric = st.selectbox("Colour the map by", ["Heatwave days / yr", "Hot days / yr",
                                                "Avg max temp (°C)", "Annual rain (mm)"])
    scale = "Blues" if "rain" in metric.lower() else THERMAL
    c1, c2 = st.columns([3, 2])
    with c1:
        try:
            fig = px.scatter_geo(dist, lat="lat", lon="lon", color=metric, hover_name="District",
                                 color_continuous_scale=scale, title=f"{metric} by district")
            fig.update_traces(marker=dict(size=13, line=dict(width=0.5, color="grey")))
            fig.update_geos(fitbounds="locations", resolution=50, showcountries=True, showsubunits=True)
        except Exception:
            fig = px.scatter(dist, x="lon", y="lat", color=metric, hover_name="District",
                             color_continuous_scale=scale, title=f"{metric} by district")
        show(fig, 460)
    with c2:
        ranked = dist.sort_values(metric, ascending=False)
        top = pd.concat([ranked.head(8), ranked.tail(8)]).sort_values(metric)
        fig = px.bar(top, x=metric, y="District", orientation="h", color=metric,
                     color_continuous_scale=scale, title="Top 8 and bottom 8")
        fig.update_layout(coloraxis_showscale=False, yaxis_title="")
        show(fig, 460)
    hi_d, lo_d = ranked.iloc[0], ranked.iloc[-1]
    cl = dist[[metric, "lat", "lon"]].corr()[metric]
    note(f"**{hi_d['District'].replace('_', ' ')}** has the highest value ({hi_d[metric]:.1f}) and "
         f"**{lo_d['District'].replace('_', ' ')}** the lowest ({lo_d[metric]:.1f}) for *{metric}*. "
         f"The districts average {dist[metric].mean():.1f} (standard deviation {dist[metric].std():.1f}). "
         f"The link with latitude is r = {cl['lat']:.2f} and with longitude r = {cl['lon']:.2f}, so "
         f"{'location explains a good part of the differences' if max(abs(cl['lat']), abs(cl['lon'])) > 0.5 else 'location alone does not fully explain the differences'}.")

# ---------------------------------------------------------------- Predict
with t_pred:
    predict_tab()

# ------------------------------------------------------------ About the Data
with t_about:
    about_tab()