# 🦠 Smart Water-Borne Disease Alert Dashboard

An ML-powered early-warning dashboard for water-borne disease risk across North-East India — case + death history, a red/yellow/green risk map, live weather, live news, a hospital locator, email alerts, and an optional AI helper bot.

## 📦 Files you need (all included)

```
health_dashboard/
├── app.py                      # the dashboard
├── generate_datasets.py        # regenerates the synthetic data below
├── requirements.txt
├── .gitignore                  # keeps .env out of git — important, see below
└── data/
    ├── NE_WaterBorne_2022_Synthetic.xlsx
    ├── NE_WaterBorne_2023_Synthetic.xlsx
    ├── NE_WaterBorne_2024_Synthetic.xlsx
    ├── NE_WaterBorne_2025_Synthetic.xlsx   ← Cases + Deaths + Outbreak_Spike columns
    ├── NE_RainfallHumidity.xlsx
    ├── NE_StateProfile.xlsx
    └── NE_Hospitals.xlsx                   ← ⚠️ synthetic placeholder data, see below
```

## 🚀 Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

First launch asks for your name + email (one-time, saved to a local `.env` file) — used as the default sender/recipient for alert emails elsewhere in the app.

## ✨ What's new in this version

- **Deaths column** added to all disease data, using illustrative case-fatality ratios per disease
- **2025 dataset** added (4 years total now: 2022–2025)
- **Outbreak spikes** — ~7% of state/disease/months get a random realistic outbreak spike, so the model has real irregularity to deal with (its accuracy score is now honestly lower, ~0.6 R² instead of an inflated ~0.98 — that's a *good* sign, not a bug)
- **Risk Zone Map fixed** — zones now compare each state/disease's prediction against *that same calendar month's* historical average (not a flat yearly average), and roll up to a state-level zone using whichever disease is riskiest. You'll now see a real mix of Green/Yellow/Red instead of everything defaulting to green
- **Disease breakdown** — the map and a detail table show *which disease* is driving each state's zone
- **Plain-language labels** — "Lag1"/"Lag2" now show as "Cases last month" / "Cases 2 months ago" in the UI, and there's a rule-based advice line under every prediction
- **🏥 Hospital Locator** — search by state or place name, see nearby hospitals (beds, doctors, medicine stock), and message one directly
- **🤖 HelperBot** — optional AI chat assistant (Groq API) for water-safety questions
- **✅ Do's & Don'ts** — its own dedicated page now, not buried in a tab
- **📖 Documentation** — rewritten in plain language, explains the model and its limits honestly
- **ℹ️ About** — generic project description, no personal info

## 🔐 About the `.env` file — please don't share it

After first run (and after setting up email alerts or HelperBot), `.env` will contain your **email, Gmail App Password, and Groq API key** in plain text. The included `.gitignore` already excludes it from git — just don't paste its contents anywhere public. To reset everything, just delete `.env` and restart the app.

## ⚠️ Two things that are demo-only — please read

1. **Hospital data is synthetic.** Names, bed counts, doctors, and `@demohospitals.example` email addresses in `NE_Hospitals.xlsx` are placeholders for the UI demo — they are **not real hospitals**. The "message hospital" button will not reach anyone until you replace this file with real, verified hospital contacts.
2. **All disease numbers are synthetic**, calibrated to *feel* realistic (real seasonal patterns, real-ish case-fatality ratios) but not sourced from actual government surveillance records. The Documentation page says this too.

## 📧 Setting up email alerts (optional)

Needs a **Gmail App Password**, not your normal password:
1. Google Account → Security → 2-Step Verification (must be on)
2. Search "App Passwords" → generate one for "Mail"
3. Paste that 16-character password into the Alerts page — it's saved locally to `.env` so you won't need to re-enter it

## 🤖 Setting up HelperBot (optional)

1. Get a free key at [console.groq.com/keys](https://console.groq.com/keys)
2. Paste it in on the HelperBot page — saved locally to `.env`

If the model listed in `app.py` (`GROQ_MODEL`) ever gets retired, check [console.groq.com/docs/models](https://console.groq.com/docs/models) for the current model name and swap it in.

## 🧠 Model & Risk Zone notes

- Features: month, state, disease, last 2 months' case counts, rainfall, humidity, sanitation/water-quality/population indicators.
- Risk ratio = predicted next month ÷ that state-disease's own historical average for the *same calendar month*. Green < 1.10× · Yellow 1.10–1.39× · Red ≥ 1.40×.
- Thresholds and CFR values are illustrative — edit the constants near the top of `app.py` / `generate_datasets.py` to change them.

## 🛠️ Tech Stack

`Python` · `Streamlit` · `scikit-learn` · `Plotly` · `Open-Meteo API` · `Google News RSS` · `Groq API` (optional)
