"""
Smart Water-Borne Disease Alert Dashboard — North-East India
==============================================================
An ML-powered early-warning dashboard for water-borne disease risk across
North-East Indian states: historical case+death data, live weather, live
news, a red/yellow/green risk map, a hospital locator, and an optional
AI helper bot.
"""
import math
import os
import xml.etree.ElementTree as ET
from datetime import datetime
from urllib.parse import quote as url_quote

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import streamlit as st
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                               RandomForestRegressor, StackingRegressor)
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Water-Borne Disease Alert System",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
MONTH_MAP = {m: i + 1 for i, m in enumerate(MONTHS)}
REV_MONTH_MAP = {v: k for k, v in MONTH_MAP.items()}

DATA_DIR = "data"
DISEASE_YEARS = [2022, 2023, 2024, 2025]
DISEASE_FILES = [(f"{DATA_DIR}/NE_WaterBorne_{y}_Synthetic.xlsx", y) for y in DISEASE_YEARS]
WEATHER_FILE = f"{DATA_DIR}/NE_RainfallHumidity.xlsx"
PROFILE_FILE = f"{DATA_DIR}/NE_StateProfile.xlsx"
HOSPITAL_FILE = f"{DATA_DIR}/NE_Hospitals.xlsx"

ENV_FILE = ".env"
GROQ_MODEL = "llama-3.1-8b-instant" 

RISK_COLORS = {"Green": "#2ecc71", "Yellow": "#f1c40f", "Red": "#e74c3c"}
RISK_ORDER = ["Green", "Yellow", "Red"]

FRIENDLY_FEATURE_NAMES = {
    "MonthNum": "Month of the year",
    "State_Code": "Which state",
    "Disease_Code": "Which disease",
    "Lag1": "Cases reported last month",
    "Lag2": "Cases reported 2 months ago",
    "Rainfall_mm": "Rainfall that month",
    "Humidity_%": "Humidity level",
    "Sanitation_Index": "State's sanitation level",
    "Water_Quality_Index": "State's water quality",
    "Population_Lakhs": "State's population size",
}

# ----------------------------------------------------------------------------
# THEME — dark / neon "hacker-techy" look
# ----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;800&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp {
  background: radial-gradient(ellipse 900px 500px at 15% -10%, rgba(0,255,106,0.06), transparent),
              #060b08;
  color: #e3ffe9;
}
h1, h2, h3 { font-family: 'JetBrains Mono', monospace !important; color: #e3ffe9; }
[data-testid="stSidebar"] { background: #081109; border-right: 1px solid #163d24; }
.metric-card {
  background: #0b1b10; border: 1px solid #163d24; border-radius: 12px;
  padding: 16px 18px; margin-bottom: 10px;
}
.zone-pill {
  display:inline-block; padding: 4px 12px; border-radius: 20px;
  font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; font-weight: 700;
}
.small-mono { font-family: 'JetBrains Mono', monospace; color: #7fd9a4; font-size: 0.82rem; }
.advice-box {
  background: #0b1b10; border-left: 3px solid #2ecc71; border-radius: 8px;
  padding: 14px 18px; margin-top: 10px;
}
hr { border-color: #163d24 !important; }
.stButton>button {
  background: #10331d; color: #7dffb0; border: 1px solid #1f6b3c; border-radius: 8px;
  font-family: 'JetBrains Mono', monospace;
}
.stButton>button:hover { background:#15452a; border-color:#2ecc71; color:#eafff2; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# .env HELPERS  (simple KEY=VALUE file, no extra dependency)
# ----------------------------------------------------------------------------
def load_env():
    data = {}
    if os.path.isfile(ENV_FILE):
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    data[k.strip()] = v.strip()
   
    try:
        for k, v in st.secrets.items():
            data[k] = v
    except Exception:
        pass 
    return data


def save_env(updates: dict):
    data = load_env()
    data.update(updates)
    with open(ENV_FILE, "w", encoding="utf-8") as f:
        for k, v in data.items():
            f.write(f"{k}={v}\n")


ENV = load_env()


# ----------------------------------------------------------------------------
# DATA LOADING  (cached — reruns only if files change)
# ----------------------------------------------------------------------------
def _file_exists(path):
    return os.path.isfile(path)


@st.cache_data(show_spinner=False)
def load_and_merge_data():
    missing = [f for f, _ in DISEASE_FILES if not _file_exists(f)]
    for extra in (WEATHER_FILE, PROFILE_FILE, HOSPITAL_FILE):
        if not _file_exists(extra):
            missing.append(extra)
    if missing:
        return None, missing

    dfs = []
    for f, y in DISEASE_FILES:
        d = pd.read_excel(f)
        d["Year"] = y
        dfs.append(d)
    disease_df = pd.concat(dfs, ignore_index=True)
    weather_df = pd.read_excel(WEATHER_FILE)
    profile_df = pd.read_excel(PROFILE_FILE)
    hospitals_df = pd.read_excel(HOSPITAL_FILE)

    disease_df["MonthNum"] = disease_df["Month"].map(MONTH_MAP)
    weather_df["MonthNum"] = weather_df["Month"].map(MONTH_MAP)

    merged = disease_df.merge(
        weather_df[["Year", "State", "MonthNum", "Rainfall_mm", "Humidity_%"]],
        on=["Year", "State", "MonthNum"], how="left",
    )
    merged = merged.merge(
        profile_df[["State", "Sanitation_Index", "Water_Quality_Index", "Population_Lakhs"]],
        on="State", how="left",
    )
    merged["Rainfall_mm"] = merged["Rainfall_mm"].fillna(merged["Rainfall_mm"].mean())
    merged["Humidity_%"] = merged["Humidity_%"].fillna(merged["Humidity_%"].mean())

    states = sorted(merged["State"].unique().tolist())
    diseases = sorted(merged["Disease"].unique().tolist())
    state_to_code = {s: i for i, s in enumerate(states)}
    disease_to_code = {d: i for i, d in enumerate(diseases)}
    merged["State_Code"] = merged["State"].map(state_to_code)
    merged["Disease_Code"] = merged["Disease"].map(disease_to_code)

    merged = merged.sort_values(["State", "Disease", "Year", "MonthNum"]).reset_index(drop=True)
    merged["Lag1"] = merged.groupby(["State", "Disease"])["Cases"].shift(1)
    merged["Lag2"] = merged.groupby(["State", "Disease"])["Cases"].shift(2)
    merged["Lag1"] = merged["Lag1"].fillna(merged["Cases"].mean())
    merged["Lag2"] = merged["Lag2"].fillna(merged["Cases"].mean())

    # empirical case-fatality ratio per disease (used to translate predicted cases -> expected deaths)
    cfr_by_disease = (
        merged.groupby("Disease").apply(
            lambda d: d["Deaths"].sum() / d["Cases"].sum() if d["Cases"].sum() > 0 else 0
        ).to_dict()
    )

    bundle = {
        "merged": merged,
        "weather_df": weather_df,
        "profile_df": profile_df,
        "hospitals_df": hospitals_df,
        "states": states,
        "diseases": diseases,
        "state_to_code": state_to_code,
        "disease_to_code": disease_to_code,
        "cfr_by_disease": cfr_by_disease,
        "feature_cols": ["MonthNum", "State_Code", "Disease_Code", "Lag1", "Lag2",
                          "Rainfall_mm", "Humidity_%", "Sanitation_Index",
                          "Water_Quality_Index", "Population_Lakhs"],
    }
    return bundle, []


# ----------------------------------------------------------------------------
# MODEL TRAINING  (cached resource — trains once per data version)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def train_model(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    base_models = [
        ("rf", RandomForestRegressor(n_estimators=200, random_state=42)),
        ("gb", GradientBoostingRegressor(n_estimators=200, random_state=42)),
        ("et", ExtraTreesRegressor(n_estimators=200, random_state=42)),
    ]
    model = StackingRegressor(estimators=base_models,
                               final_estimator=RandomForestRegressor(random_state=42))
    model.fit(X_train, y_train)
    pred_test = model.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, pred_test),
        "mae": mean_absolute_error(y_test, pred_test),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    rf = model.named_estimators_["rf"]
    importances = dict(sorted(zip(X.columns, rf.feature_importances_), key=lambda t: -t[1]))
    return model, metrics, importances


# ----------------------------------------------------------------------------
# FORECASTING
# ----------------------------------------------------------------------------
def forecast_12_months(model, feature_cols, merged, weather_df, profile_df,
                        state, disease, state_to_code, disease_to_code, cfr_by_disease):
    sub = merged[(merged.State == state) & (merged.Disease == disease)].sort_values(["Year", "MonthNum"])
    if sub.empty:
        return pd.DataFrame(columns=["Month", "Predicted_Cases", "Predicted_Deaths"])

    last = sub.iloc[-1]
    lag1, lag2 = last["Cases"], sub.iloc[-2]["Cases"] if len(sub) > 1 else last["Cases"]
    s_code, d_code = state_to_code[state], disease_to_code[disease]
    profile_row = profile_df[profile_df.State == state].iloc[0]
    state_weather = weather_df[weather_df.State == state]
    cfr = cfr_by_disease.get(disease, 0.005)

    rows = []
    for m in range(1, 13):
        month_weather = state_weather[state_weather.MonthNum == m]
        avg_rain = month_weather["Rainfall_mm"].mean() if not month_weather.empty else state_weather["Rainfall_mm"].mean()
        avg_hum = month_weather["Humidity_%"].mean() if not month_weather.empty else state_weather["Humidity_%"].mean()

        x_row = pd.DataFrame([{
            "MonthNum": m, "State_Code": s_code, "Disease_Code": d_code,
            "Lag1": lag1, "Lag2": lag2, "Rainfall_mm": avg_rain, "Humidity_%": avg_hum,
            "Sanitation_Index": profile_row["Sanitation_Index"],
            "Water_Quality_Index": profile_row["Water_Quality_Index"],
            "Population_Lakhs": profile_row["Population_Lakhs"],
        }])[feature_cols]

        pred = max(0, model.predict(x_row)[0])
        rows.append({
            "Month": REV_MONTH_MAP[m],
            "Predicted_Cases": round(pred),
            "Predicted_Deaths": round(pred * cfr, 1),
        })
        lag2, lag1 = lag1, pred

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# RISK ZONE CALCULATION
# (compares next month's prediction against that SAME CALENDAR MONTH's
#  historical average, per state AND per disease — a fair seasonal comparison)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_risk_zones(_model, feature_cols, merged, weather_df, profile_df,
                        states, diseases, state_to_code, disease_to_code, cfr_by_disease):
    detail_records = []
    for state in states:
        for disease in diseases:
            sub = merged[(merged.State == state) & (merged.Disease == disease)]
            fc = forecast_12_months(_model, feature_cols, merged, weather_df, profile_df,
                                     state, disease, state_to_code, disease_to_code, cfr_by_disease)
            if fc.empty:
                continue
            next_month_name = fc.iloc[0]["Month"]
            next_month_num = MONTH_MAP[next_month_name]
            predicted_cases = fc.iloc[0]["Predicted_Cases"]
            predicted_deaths = fc.iloc[0]["Predicted_Deaths"]

            same_month_hist = sub[sub.MonthNum == next_month_num]["Cases"]
            hist_avg = same_month_hist.mean() if len(same_month_hist) else sub["Cases"].mean()
            ratio = predicted_cases / hist_avg if hist_avg > 0 else 1.0

            if ratio >= 1.4:
                zone = "Red"
            elif ratio >= 1.1:
                zone = "Yellow"
            else:
                zone = "Green"

            detail_records.append({
                "State": state, "Disease": disease, "Next_Month": next_month_name,
                "Predicted_Cases": predicted_cases, "Predicted_Deaths": predicted_deaths,
                "Historical_Same_Month_Avg": round(hist_avg, 1),
                "Risk_Ratio": round(ratio, 2), "Zone": zone,
            })

    detail_df = pd.DataFrame(detail_records)

    # state-level rollup = worst zone among that state's diseases (max ratio)
    state_records = []
    for state in states:
        state_detail = detail_df[detail_df.State == state]
        if state_detail.empty:
            continue
        worst = state_detail.loc[state_detail["Risk_Ratio"].idxmax()]
        profile_row = profile_df[profile_df.State == state].iloc[0]
        state_records.append({
            "State": state,
            "Zone": worst["Zone"],
            "Driving_Disease": worst["Disease"],
            "Risk_Ratio": worst["Risk_Ratio"],
            "Predicted_Cases_Total": round(state_detail["Predicted_Cases"].sum()),
            "Predicted_Deaths_Total": round(state_detail["Predicted_Deaths"].sum(), 1),
            "Latitude": profile_row["Latitude"],
            "Longitude": profile_row["Longitude"],
            "Sanitation_Index": profile_row["Sanitation_Index"],
        })
    state_df = pd.DataFrame(state_records)
    return state_df, detail_df


# ----------------------------------------------------------------------------
# ADVICE ENGINE  (rule-based — always available, no external API needed)
# ----------------------------------------------------------------------------
ADVICE_BY_DISEASE = {
    "Diarrhea": "Prioritise oral rehydration (ORS) availability and safe drinking water messaging in this area.",
    "Cholera": "Cholera can escalate fast — alert the nearest health facility, check chlorination of water sources, and watch for clusters of cases.",
    "Typhoid": "Encourage vaccination drives where available and inspect food/water vendors near the affected areas.",
    "Hepatitis A": "Push sanitation and safe-food messaging; Hepatitis A spreads through contaminated food and water.",
    "Dysentery": "Check for contaminated water sources and ensure antibiotics/ORS stock at local facilities if cases rise.",
}


def get_advice(disease, zone):
    base = ADVICE_BY_DISEASE.get(disease, "Monitor water and sanitation conditions closely.")
    if zone == "Red":
        return f"🔴 High risk: {base} Consider notifying the district health office proactively."
    elif zone == "Yellow":
        return f"🟡 Elevated risk: {base} Keep monitoring — no need to escalate yet."
    return f"🟢 Normal range: {base}"


# ----------------------------------------------------------------------------
# LIVE WEATHER  (Open-Meteo — free, no API key required)
# ----------------------------------------------------------------------------
WEATHER_CODES = {
    0: ("Clear sky", "☀️"), 1: ("Mainly clear", "🌤️"), 2: ("Partly cloudy", "⛅"),
    3: ("Overcast", "☁️"), 45: ("Fog", "🌫️"), 48: ("Depositing fog", "🌫️"),
    51: ("Light drizzle", "🌦️"), 53: ("Drizzle", "🌦️"), 55: ("Dense drizzle", "🌧️"),
    61: ("Light rain", "🌧️"), 63: ("Rain", "🌧️"), 65: ("Heavy rain", "⛈️"),
    80: ("Rain showers", "🌦️"), 81: ("Rain showers", "🌧️"), 82: ("Violent showers", "⛈️"),
    95: ("Thunderstorm", "⛈️"), 96: ("Thunderstorm + hail", "⛈️"), 99: ("Severe thunderstorm", "⛈️"),
}


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_live_weather(lat, lon):
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
                "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min",
                "forecast_days": 7, "timezone": "auto",
            },
            timeout=8,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=3600, show_spinner=False)
def geocode_place(place_name):
    """Free geocoding via Open-Meteo — no API key required."""
    try:
        resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": place_name, "count": 5, "language": "en", "format": "json"},
            timeout=8,
        )
        resp.raise_for_status()
        return resp.json().get("results", [])
    except Exception:
        return []


# ----------------------------------------------------------------------------
# LIVE NEWS  (Google News RSS — free, no API key required)
# ----------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_news(query, max_items=8):
    try:
        url = f"https://news.google.com/rss/search?q={url_quote(query)}&hl=en-IN&gl=IN&ceid=IN:en"
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        items = []
        for item in root.findall(".//item")[:max_items]:
            items.append({
                "title": item.findtext("title", ""),
                "link": item.findtext("link", ""),
                "pubDate": item.findtext("pubDate", ""),
                "source": item.findtext("source", ""),
            })
        return items, None
    except Exception as e:
        return [], str(e)


# ----------------------------------------------------------------------------
# EMAIL  (SMTP — user supplies their own credentials)
# ----------------------------------------------------------------------------
def send_email(sender_email, app_password, recipient_email, subject, body):
    import smtplib
    from email.mime.text import MIMEText
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = recipient_email
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=10) as server:
            server.starttls()
            server.login(sender_email, app_password)
            server.sendmail(sender_email, [recipient_email], msg.as_string())
        return True, "Email sent successfully."
    except Exception as e:
        return False, f"Could not send email: {e}"


# ----------------------------------------------------------------------------
# HOSPITAL LOCATOR HELPERS
# ----------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def nearest_hospitals(hospitals_df, lat, lon, top_n=10):
    df = hospitals_df.copy()
    df["Distance_km"] = df.apply(lambda r: haversine_km(lat, lon, r["Latitude"], r["Longitude"]), axis=1)
    return df.sort_values("Distance_km").head(top_n)


# ----------------------------------------------------------------------------
# GROQ HELPERBOT  (optional — needs the user's own free Groq API key)
# ----------------------------------------------------------------------------
HELPERBOT_SYSTEM_PROMPT = (
    "You are HelperBot, an assistant embedded in a water-borne disease early-warning "
    "dashboard for North-East India. Answer questions about water-borne disease "
    "prevention, hygiene, sanitation, how to read the dashboard's risk zones/predictions, "
    "and general public-health guidance. Keep answers short and in plain language for "
    "the general public. You are not a doctor — for personal medical concerns, always "
    "tell the person to consult a qualified healthcare professional. Do not answer "
    "questions unrelated to health, water safety, or this dashboard."
)


def ask_helperbot(api_key, user_message, history=None):
    try:
        messages = [{"role": "system", "content": HELPERBOT_SYSTEM_PROMPT}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": GROQ_MODEL, "messages": messages, "temperature": 0.4, "max_tokens": 400},
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"], None
    except Exception as e:
        return None, str(e)


# ----------------------------------------------------------------------------
# FIRST-RUN SETUP  (asks for name + email once, saves locally to .env)
# ----------------------------------------------------------------------------
if not ENV.get("USER_NAME") or not ENV.get("USER_EMAIL"):
    st.title("🦠 Welcome")
    st.caption("Quick one-time setup before you start.")
    with st.form("first_run_setup"):
        name_in = st.text_input("Your name")
        email_in = st.text_input("Your email")
        st.caption("Saved locally in a `.env` file on this machine only — used as the default "
                   "sender/recipient for the alert emails elsewhere in the app. Nothing is sent anywhere else.")
        go = st.form_submit_button("Continue →")
        if go:
            if name_in and email_in:
                save_env({"USER_NAME": name_in, "USER_EMAIL": email_in})
                st.rerun()
            else:
                st.warning("Please fill in both fields.")
    st.stop()

USER_NAME = ENV.get("USER_NAME", "")
USER_EMAIL = ENV.get("USER_EMAIL", "")

# ----------------------------------------------------------------------------
# LOAD DATA + TRAIN MODEL
# ----------------------------------------------------------------------------
bundle, missing_files = load_and_merge_data()

if bundle is None:
    st.title("🦠 Water-Borne Disease Alert Dashboard")
    st.error("🚫 Required data files are missing:")
    for f in missing_files:
        st.code(f)
    st.info("Run `python generate_datasets.py` to create the sample synthetic datasets. See README.md for details.")
    st.stop()

merged = bundle["merged"]
weather_df = bundle["weather_df"]
profile_df = bundle["profile_df"]
hospitals_df = bundle["hospitals_df"]
states = bundle["states"]
diseases = bundle["diseases"]
state_to_code = bundle["state_to_code"]
disease_to_code = bundle["disease_to_code"]
cfr_by_disease = bundle["cfr_by_disease"]
feature_cols = bundle["feature_cols"]

X = merged[feature_cols]
y = merged["Cases"]
model, metrics, importances = train_model(X, y)
risk_state_df, risk_detail_df = compute_risk_zones(
    model, feature_cols, merged, weather_df, profile_df,
    states, diseases, state_to_code, disease_to_code, cfr_by_disease,
)

# ----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------------
st.sidebar.markdown("### 🦠 WBD Alert System")
st.sidebar.caption(f"Signed in as {USER_NAME}")

red_count = int((risk_state_df.Zone == "Red").sum())
yellow_count = int((risk_state_df.Zone == "Yellow").sum())
if red_count:
    st.sidebar.error(f"🔴 {red_count} state(s) in RED zone")
if yellow_count:
    st.sidebar.warning(f"🟡 {yellow_count} state(s) in YELLOW zone")
if red_count == 0 and yellow_count == 0:
    st.sidebar.success("🟢 All states nominal")

page = st.sidebar.radio("Navigate", [
    " Overview", " Disease Prediction", " Risk Zone Map",
    " Live Weather", " Live News", " Hospital Locator",
    " HelperBot (AI)", " Alerts & Notifications",
    " Do's & Don'ts", " Documentation", " About",
])
st.sidebar.markdown("---")
st.sidebar.markdown(
    f'<span class="small-mono">Data: synthetic demo · {len(merged)} records<br>'
    f'Updated {datetime.now().strftime("%b %Y")}</span>',
    unsafe_allow_html=True,
)

# ============================================================================
# PAGE: OVERVIEW
# ============================================================================
if page == " Overview":
    st.title("🦠 Water-Borne Disease Alert Dashboard")
    st.caption("North-East India · ML-powered early warning system · synthetic demo data")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("States Monitored", len(states))
    c2.metric("Diseases Tracked", len(diseases))
    c3.metric("Total Cases (all years)", f"{int(merged['Cases'].sum()):,}")
    c4.metric("Total Deaths (all years)", f"{int(merged['Deaths'].sum()):,}")
    c5.metric("🔴 Red Zone States", red_count)

    st.markdown("---")
    col_a, col_b = st.columns([1.3, 1])
    with col_a:
        st.subheader("Total Cases by State (All-Time)")
        totals = merged.groupby("State")["Cases"].sum().sort_values(ascending=True).reset_index()
        fig = px.bar(totals, x="Cases", y="State", orientation="h",
                     color="Cases", color_continuous_scale=["#2ecc71", "#f1c40f", "#e74c3c"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e3ffe9", height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        st.subheader("Cases by Disease")
        dtotals = merged.groupby("Disease")["Cases"].sum().reset_index()
        fig2 = px.pie(dtotals, names="Disease", values="Cases", hole=0.45)
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e3ffe9", height=380)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Monthly Trend — Cases & Deaths (All States, All Diseases)")
    trend = merged.groupby(["Year", "MonthNum"])[["Cases", "Deaths"]].sum().reset_index()
    trend["Period"] = trend["Year"].astype(str) + "-" + trend["MonthNum"].astype(str).str.zfill(2)
    trend = trend.sort_values(["Year", "MonthNum"])
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=trend["Period"], y=trend["Cases"], mode="lines", name="Cases",
                               line=dict(color="#2ecc71", width=2)))
    fig3.add_trace(go.Bar(x=trend["Period"], y=trend["Deaths"], name="Deaths",
                           marker_color="#e74c3c", yaxis="y2", opacity=0.6))
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e3ffe9",
        height=340, xaxis_tickangle=-45,
        yaxis=dict(title="Cases"), yaxis2=dict(title="Deaths", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig3, use_container_width=True)
    n_spikes = int(merged["Outbreak_Spike"].sum()) if "Outbreak_Spike" in merged.columns else 0
    st.caption(f" {n_spikes} localized outbreak-spike events flagged across the full dataset "
               f"(sudden jumps well above a state/disease's normal pattern).")


# ============================================================================
# PAGE: DISEASE PREDICTION
# ============================================================================
elif page == " Disease Prediction":
    st.title("📈 Disease Prediction")
    st.caption("12-month forward forecast using a stacked machine-learning model")

    col1, col2 = st.columns(2)
    selected_disease = col1.selectbox("Choose Disease", diseases)
    selected_state = col2.selectbox("Choose State", states)

    forecast_df = forecast_12_months(model, feature_cols, merged, weather_df, profile_df,
                                      selected_state, selected_disease, state_to_code,
                                      disease_to_code, cfr_by_disease)

    total_cases = int(forecast_df["Predicted_Cases"].sum())
    total_deaths = round(forecast_df["Predicted_Deaths"].sum(), 1)
    m1, m2 = st.columns(2)
    m1.success(f"💡 Estimated **{selected_disease}** cases in **{selected_state}**, next 12 months: **{total_cases:,}**")
    m2.warning(f"⚠️ Estimated deaths over the same period: **{total_deaths}**")

    fig = px.bar(forecast_df, x="Month", y="Predicted_Cases",
                 color="Predicted_Cases", color_continuous_scale=["#2ecc71", "#f1c40f", "#e74c3c"])
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font_color="#e3ffe9", height=380, showlegend=False,
                       title=f"{selected_state} — {selected_disease} (Next 12 Months)")
    st.plotly_chart(fig, use_container_width=True)

    # advice box, tied to this disease/state's actual computed risk zone
    match = risk_detail_df[(risk_detail_df.State == selected_state) & (risk_detail_df.Disease == selected_disease)]
    zone_here = match.iloc[0]["Zone"] if not match.empty else "Green"
    st.markdown(f'<div class="advice-box"> <b>Advice:</b> {get_advice(selected_disease, zone_here)}</div>',
                unsafe_allow_html=True)

    st.markdown("---")
    col_dl, col_metrics = st.columns([1, 1.3])
    with col_dl:
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)
        csv_bytes = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download forecast as CSV", csv_bytes,
                            file_name=f"{selected_state}_{selected_disease}_forecast.csv", mime="text/csv")
    with col_metrics:
        st.markdown("##### Model performance (on data the model hasn't seen)")
        mc1, mc2 = st.columns(2)
        mc1.metric("Accuracy score (R²)", f"{metrics['r2']:.2f}")
        mc2.metric("Average error (MAE)", f"{metrics['mae']:.0f} cases")
        st.caption("R² closer to 1.0 = better fit. MAE = typical prediction error in raw case counts. "
                   "Sudden outbreak spikes are, realistically, hard for any model to predict in advance.")

        st.markdown("##### What drives this prediction most?")
        imp_df = pd.DataFrame(list(importances.items()), columns=["Feature", "Importance"]).head(6)
        imp_df["Feature"] = imp_df["Feature"].map(lambda f: FRIENDLY_FEATURE_NAMES.get(f, f))
        fig_imp = px.bar(imp_df.sort_values("Importance"), x="Importance", y="Feature", orientation="h")
        fig_imp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#e3ffe9", height=260, showlegend=False)
        st.plotly_chart(fig_imp, use_container_width=True)
        st.caption("In plain terms: how many cases were reported in the last couple of months is usually "
                   "the single biggest clue for what happens next — rainfall and humidity add a secondary push.")

    st.caption("Note: this 12-month forecast assumes future weather follows the historical seasonal average "
               "for each month — it does not pull real future rainfall forecasts (only the Live Weather page does that).")


# ============================================================================
# PAGE: RISK ZONE MAP
# ============================================================================
elif page == " Risk Zone Map":
    st.title(" Risk Zone Map")
    st.caption("Each state's predicted next month vs. its own historical average for that same calendar month")

    legend_cols = st.columns(3)
    legend_cols[0].markdown('<span class="zone-pill" style="background:#1d4a2c;color:#2ecc71;">🟢 GREEN — Normal</span>', unsafe_allow_html=True)
    legend_cols[1].markdown('<span class="zone-pill" style="background:#4a3f14;color:#f1c40f;">🟡 YELLOW — Elevated</span>', unsafe_allow_html=True)
    legend_cols[2].markdown('<span class="zone-pill" style="background:#4a1d1d;color:#e74c3c;">🔴 RED — High Risk</span>', unsafe_allow_html=True)

    fig_map = px.scatter_mapbox(
        risk_state_df, lat="Latitude", lon="Longitude", color="Zone",
        color_discrete_map=RISK_COLORS, size="Predicted_Cases_Total",
        size_max=40, zoom=4.6, hover_name="State",
        hover_data={"Driving_Disease": True, "Predicted_Cases_Total": True,
                    "Predicted_Deaths_Total": True, "Risk_Ratio": True,
                    "Latitude": False, "Longitude": False, "Zone": False},
        mapbox_style="open-street-map", height=540,
    )
    fig_map.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e3ffe9",
                           margin=dict(l=0, r=0, t=10, b=0),
                           legend=dict(bgcolor="rgba(0,0,0,0.4)"))
    st.plotly_chart(fig_map, use_container_width=True)

    st.subheader("Which disease is driving each state's zone?")
    display_df = risk_state_df[["State", "Zone", "Driving_Disease", "Risk_Ratio",
                                 "Predicted_Cases_Total", "Predicted_Deaths_Total"]].sort_values(
        "Risk_Ratio", ascending=False)

    def _zone_style(row):
        return [f"background-color: {RISK_COLORS[row.Zone]}33" for _ in row]

    st.dataframe(display_df.style.apply(_zone_style, axis=1), use_container_width=True, hide_index=True)

    st.markdown("##### Full disease-by-disease breakdown")
    pick_state = st.selectbox("Pick a state to see all 5 diseases individually", states)
    state_detail = risk_detail_df[risk_detail_df.State == pick_state][
        ["Disease", "Zone", "Risk_Ratio", "Predicted_Cases", "Predicted_Deaths", "Historical_Same_Month_Avg"]
    ].sort_values("Risk_Ratio", ascending=False)
    st.dataframe(state_detail.style.apply(_zone_style, axis=1), use_container_width=True, hide_index=True)

    st.caption("Zone thresholds: Green < 1.10× · Yellow 1.10–1.39× · Red ≥ 1.40× that disease's own historical "
               "average for the same calendar month (e.g. this July vs. past Julys) — a fair, season-matched comparison.")


# ============================================================================
# PAGE: LIVE WEATHER
# ============================================================================
elif page == " Live Weather":
    st.title(" Live Weather")
    st.caption("Real-time conditions via the Open-Meteo API (free, no API key required)")

    selected_state_w = st.selectbox("Select State", states, key="weather_state")
    prow = profile_df[profile_df.State == selected_state_w].iloc[0]
    lat, lon = prow["Latitude"], prow["Longitude"]

    with st.spinner("Fetching live weather..."):
        live = fetch_live_weather(lat, lon)

    if live and "error" not in live:
        cur = live.get("current", {})
        code = cur.get("weather_code", 0)
        desc, emoji = WEATHER_CODES.get(code, ("Unknown", "🌡️"))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"{emoji} Condition", desc)
        c2.metric("Temperature", f"{cur.get('temperature_2m', '–')} °C")
        c3.metric("Humidity", f"{cur.get('relative_humidity_2m', '–')}%")
        c4.metric("Wind", f"{cur.get('wind_speed_10m', '–')} km/h")

        daily = live.get("daily", {})
        if daily:
            daily_df = pd.DataFrame({
                "Date": daily.get("time", []),
                "Rain (mm)": daily.get("precipitation_sum", []),
                "Max Temp (°C)": daily.get("temperature_2m_max", []),
                "Min Temp (°C)": daily.get("temperature_2m_min", []),
            })
            fig_w = go.Figure()
            fig_w.add_trace(go.Bar(x=daily_df["Date"], y=daily_df["Rain (mm)"],
                                    name="Rain (mm)", marker_color="#3498db"))
            fig_w.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                 font_color="#e3ffe9", height=300, title="7-Day Precipitation Forecast")
            st.plotly_chart(fig_w, use_container_width=True)
            st.dataframe(daily_df, use_container_width=True, hide_index=True)
    else:
        err = live.get("error") if live else "no response"
        st.warning(f"Could not fetch live weather right now ({err}). Needs internet access from wherever "
                   f"the app is running — showing historical averages below instead.")

    st.markdown("---")
    st.subheader(f"Historical Rainfall & Humidity Trend — {selected_state_w} (2022–2025)")
    hist = weather_df[weather_df.State == selected_state_w].sort_values(["Year", "MonthNum"]).copy()
    hist["Period"] = hist["Year"].astype(str) + "-" + hist["Month"]
    fig_h = go.Figure()
    fig_h.add_trace(go.Scatter(x=hist["Period"], y=hist["Rainfall_mm"], name="Rainfall (mm)",
                                line=dict(color="#3498db"), yaxis="y1"))
    fig_h.add_trace(go.Scatter(x=hist["Period"], y=hist["Humidity_%"], name="Humidity (%)",
                                line=dict(color="#f39c12"), yaxis="y2"))
    fig_h.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e3ffe9",
        height=360, xaxis_tickangle=-45,
        yaxis=dict(title="Rainfall (mm)"), yaxis2=dict(title="Humidity (%)", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig_h, use_container_width=True)


# ============================================================================
# PAGE: LIVE NEWS
# ============================================================================
elif page == " Live News":
    st.title(" Live Health News")
    st.caption("Real-time headlines via Google News RSS (free, no API key required)")

    default_query = "waterborne disease outbreak India"
    query = st.text_input("Search topic", value=default_query)
    if st.button("🔍 Fetch latest news"):
        with st.spinner("Fetching latest headlines..."):
            items, err = fetch_news(query)
        if err:
            st.warning(f"Could not fetch news right now ({err}). Needs internet access from wherever the app is running.")
        elif not items:
            st.info("No results found for this query.")
        else:
            for it in items:
                st.markdown(f"**[{it['title']}]({it['link']})**")
                st.caption(f"{it['source']} · {it['pubDate']}")
                st.markdown("---")
    else:
        st.info("Click the button above to fetch live headlines for this topic.")


# ============================================================================
# PAGE: HOSPITAL LOCATOR
# ============================================================================
elif page == " Hospital Locator":
    st.title(" Hospital Locator")
    st.warning("⚠️ **Demo data notice:** hospital names, bed counts, staff numbers, and contact details below "
               "are all **synthetically generated placeholders** for this portfolio project — they are not "
               "real hospital records. Replace `data/NE_Hospitals.xlsx` with verified real hospital data "
               "before using this for anything real.")

    find_mode = st.radio("Find hospitals by:", ["My State", "A place name"], horizontal=True)

    if find_mode == "My State":
        loc_state = st.selectbox("Select State", states, key="hosp_state")
        prow = profile_df[profile_df.State == loc_state].iloc[0]
        lat, lon = prow["Latitude"], prow["Longitude"]
        st.caption(f"Using {prow['Capital']} as the reference point.")
    else:
        place = st.text_input("Type a place name (e.g. 'Dibrugarh' or 'Shillong')")
        lat, lon = None, None
        if place:
            with st.spinner("Looking up location..."):
                results = geocode_place(place)
            if results:
                options = {f"{r['name']}, {r.get('admin1', '')} ({r['latitude']:.2f},{r['longitude']:.2f})": r for r in results}
                choice = st.selectbox("Did you mean:", list(options.keys()))
                lat, lon = options[choice]["latitude"], options[choice]["longitude"]
            else:
                st.info("No matches found yet, or no internet access to look it up.")

    if lat is not None and lon is not None:
        top_n = st.slider("How many nearby hospitals to show", 3, 20, 8)
        nearby = nearest_hospitals(hospitals_df, lat, lon, top_n=top_n)

        for _, h in nearby.iterrows():
            with st.container():
                st.markdown(f"#### {h['Hospital_Name']} — {h['District']}, {h['State']}")
                cc1, cc2, cc3, cc4, cc5 = st.columns(5)
                cc1.metric("Distance", f"{h['Distance_km']:.1f} km")
                cc2.metric("Beds available", f"{h['Available_Beds']}/{h['Total_Beds']}")
                cc3.metric("Doctors on staff", h["Doctors_On_Staff"])
                cc4.metric("Medicine stock", h["Medicine_Stock_Status"])
                cc5.metric("ORS/IV fluids", " Yes" if h["ORS_IV_Fluids_Available"] else "❌ No")
                st.caption(f"{h['Type']} facility · 📞 {h['Contact_Phone']} · ✉️ {h['Contact_Email']}")

                mailto = f"mailto:{h['Contact_Email']}?subject=Inquiry%20from%20WBD%20Dashboard&body=Hello%2C%0D%0A%0D%0A"
                mc1, mc2 = st.columns([1, 1])
                mc1.link_button("✉️ Message via your email app", mailto)
                with mc2.popover(" Send from dashboard instead"):
                    st.caption("Uses your saved Gmail (Alerts page) to send directly.")
                    subj = st.text_input("Subject", value="Inquiry from WBD Dashboard", key=f"subj_{h['Hospital_ID']}")
                    body_msg = st.text_area("Message", key=f"body_{h['Hospital_ID']}")
                    sender_gmail = ENV.get("ALERT_SENDER_EMAIL", USER_EMAIL)
                    app_pw_saved = ENV.get("ALERT_SENDER_APP_PASSWORD", "")
                    if st.button("Send", key=f"send_{h['Hospital_ID']}"):
                        if not app_pw_saved:
                            st.error("No saved Gmail App Password yet — set one up on the Alerts & Notifications page first.")
                        else:
                            ok, msg = send_email(sender_gmail, app_pw_saved, h["Contact_Email"], subj, body_msg)
                            if ok:
                                st.success(msg)
                            else:
                                st.error(msg)  
                st.markdown("---")
    else:
        st.info("Choose a state or search a place name above to see nearby hospitals.")


# ============================================================================
# PAGE: HELPERBOT (GROQ AI)
# ============================================================================
elif page == " HelperBot (AI)":
    st.title(" HelperBot")
    st.caption("Ask questions about water safety, hygiene, or how to read this dashboard — answered by an AI assistant.")

    groq_key = ENV.get("GROQ_API_KEY", "")
    if not groq_key:
        st.info("HelperBot needs a free Groq API key to work. Get one at "
                "[console.groq.com](https://console.groq.com/keys), then paste it below (saved locally only).")
        key_in = st.text_input("Groq API Key", type="password")
        if st.button("Save Key"):
            if key_in:
                save_env({"GROQ_API_KEY": key_in})
                st.rerun()
        st.stop()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Ask HelperBot something...")
    if user_q:
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, err = ask_helperbot(groq_key, user_q, st.session_state.chat_history[:-1])
            if err:
                st.error(f"HelperBot couldn't respond right now: {err}")
            else:
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

    st.caption("HelperBot is not a doctor — for personal medical concerns, always consult a qualified healthcare professional.")


# ============================================================================
# PAGE: ALERTS & NOTIFICATIONS
# ============================================================================
elif page == " Alerts & Notifications":
    app_pw = ENV.get("ALERT_SENDER_APP_PASSWORD", "")
    st.title(" Alerts & Notifications")

    red_states = risk_state_df[risk_state_df.Zone == "Red"]
    yellow_states = risk_state_df[risk_state_df.Zone == "Yellow"]

    if not red_states.empty:
        for _, r in red_states.iterrows():
            st.error(f"🔴 **{r.State}** — {r.Driving_Disease} is the main driver "
                      f"({r.Risk_Ratio}× normal for this time of year)")
            st.toast(f"🔴 High risk detected in {r.State}", icon="🚨")
    if not yellow_states.empty:
        for _, r in yellow_states.iterrows():
            st.warning(f"🟡 **{r.State}** — {r.Driving_Disease} is elevated "
                        f"({r.Risk_Ratio}× normal for this time of year)")
    if red_states.empty and yellow_states.empty:
        st.success("🟢 All states currently within normal range. No active alerts.")

    st.markdown("---")
    st.subheader("📧 Email Alert Subscription")
    st.caption("Sends a summary of current Red/Yellow zone states using your own Gmail account.")

    with st.form("email_alert_form"):
        st.info("Uses Gmail SMTP. App Password is read directly from your `.env` file "
                "(`ALERT_SENDER_APP_PASSWORD`) — no need to enter it here.")

        sender = st.text_input("Your Gmail address", value=ENV.get("ALERT_SENDER_EMAIL", USER_EMAIL))
        recipient = st.text_input("Send alert to (email)", value=USER_EMAIL)

        submitted = st.form_submit_button("Send Alert Now")

        app_pw = ENV.get("ALERT_SENDER_APP_PASSWORD", "")
      
        if submitted:
            if not app_pw:
                st.error("ALERT_SENDER_APP_PASSWORD not found in .env file, add first.")
            elif not (sender and recipient):
                st.warning("Please fill in sender and recipient email.")
            else:
                if sender != ENV.get("ALERT_SENDER_EMAIL"):
                    save_env({"ALERT_SENDER_EMAIL": sender})
                  
                summary_lines = [
                    f"{r.State}: {r.Zone} zone — {r.Driving_Disease} driving it ({r.Risk_Ratio}x normal)"
                    for _, r in risk_state_df.sort_values("Risk_Ratio", ascending=False).iterrows()
                ]
                body = "Water-Borne Disease Alert Summary\n\n" + "\n".join(summary_lines)
                ok, msg = send_email(sender, app_pw, recipient, "Water-Borne Disease Alert Summary", body)
                if ok:
                  st.success(msg)
                else:
                  st.error(msg)
              
# ============================================================================
# PAGE: DO'S & DON'TS
# ============================================================================
elif page == " Do's & Don'ts":
    st.title(" Do's & Don'ts")
    st.caption("Simple, practical steps for preventing water-borne disease — for anyone, no medical background needed.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### ✅ Do's")
        st.markdown("""
- **Boil or filter** drinking water, especially during and after heavy rainfall or flooding
- **Wash hands** with soap before eating, before cooking, and after using the toilet
- **Use ORS** (Oral Rehydration Solution) at the very first sign of diarrhea or vomiting
- **Keep stored water covered** and containers clean at all times
- **Dispose of waste and stagnant water** properly — don't let it collect near homes
- **Seek medical care promptly** if vomiting, diarrhea, or fever lasts more than a day
- **Follow local health department advisories** during outbreak alerts in your area
- **Wash fruits & vegetables** thoroughly before eating, especially raw ones
""")
    with c2:
        st.markdown("### ❌ Don'ts")
        st.markdown("""
- **Don't drink untreated water** from rivers, ponds, or open/unprotected wells
- **Don't ignore symptoms** that last more than a day or two — get checked
- **Don't self-medicate with antibiotics** without medical advice
- **Don't let children play** in or near stagnant flood water
- **Don't store water uncovered** near toilets or waste-disposal areas
- **Don't dispose of sewage** near drinking-water sources like wells or rivers
- **Don't delay reporting** suspected outbreaks to your local health authorities
- **Don't share utensils** with someone showing symptoms of a stomach illness
""")

    st.markdown("---")
    st.info("💡 **Why this matters:** most water-borne diseases spread through contaminated water or food, "
            "and simple hygiene habits prevent the large majority of cases. During monsoon season, risk "
            "rises sharply — that's exactly what this dashboard's Risk Zone Map tries to flag early.")


# ============================================================================
# PAGE: DOCUMENTATION
# ============================================================================
elif page == " Documentation":
    st.title(" Documentation")
    st.caption("How this dashboard works, in plain language.")

    st.markdown("""
### What this dashboard does
It looks at past patterns of water-borne disease cases (and deaths) across 8 North-East Indian states, 
alongside rainfall and humidity, and uses that history to **estimate** what next month might look like 
for each state and disease. States where the estimate is much higher than what's normal for that time 
of year get flagged Yellow or Red.

### Where the data comes from
- **Disease case & death data** (2022–2025) — synthetic, but built to follow real seasonal monsoon 
  patterns and realistic case-fatality ratios for diseases like cholera, typhoid, and diarrhea.
- **Weather data** — synthetic, following the real monsoon rainfall curve for North-East India.
- **State profile** — synthetic sanitation, water-quality, and population indicators per state.
- **Hospital directory** — synthetic placeholder data (see the disclaimer on the Hospital Locator page).
- **Live weather** — genuinely real-time, from the free [Open-Meteo](https://open-meteo.com) service.
- **Live news** — genuinely real-time headlines from Google News.

### How the prediction works (no jargon version)
The model mostly looks at **how many cases were reported in the last month or two** — that's usually the 
strongest clue for what happens next. It also factors in rainfall, humidity, and how good a state's 
sanitation generally is. It was tested against data it hadn't seen before, so the accuracy numbers you see 
on the Disease Prediction page are a fair (not inflated) estimate of how good it actually is.

### What "Risk Ratio" means
For a given state and disease: **(predicted cases next month) ÷ (that state's own historical average for 
the same month in past years)**. A ratio of 1.0 means "right on the usual pattern." Above 1.4 means 
"much higher than usual for this time of year" → Red.

### Limits, honestly
- This uses **synthetic demo data**, not real government surveillance records.
- The 12-month forecast assumes future weather matches the historical seasonal average — it does not 
  know about an actual real cyclone or heatwave coming next year.
- Small, sudden outbreaks (a contaminated well, a single event) are inherently hard for any model to 
  predict before they happen — that's true of real epidemiology too, not just this demo.
""")
    st.warning("**This is a demonstration / portfolio project**, not a substitute for official public "
               "health surveillance systems (such as India's Integrated Disease Surveillance Programme). "
               "It must not be used for real outbreak response, clinical decisions, or public health policy. "
               "For real health concerns, consult a qualified medical professional or your local health department.")


# ============================================================================
# PAGE: ABOUT
# ============================================================================
elif page == " About":
    st.title(" About This Project")
    st.markdown("""
**Smart Water-Borne Disease Alert Dashboard** — a machine-learning-powered early-warning concept for 
water-borne disease risk across North-East India, built as a demonstration project.

##### What's inside
Historical case/death modelling · a red/yellow/green risk map · live weather · live news · a hospital 
locator · an optional AI helper bot · email alerts.

##### Tech Stack
`Python` · `Streamlit` · `scikit-learn` (Stacking Ensemble) · `Plotly` · `Open-Meteo API` · 
`Google News RSS` · `Groq API` (optional)

##### Data Disclaimer
All disease, weather, and hospital datasets are **synthetically generated** for demonstration purposes.
""")
