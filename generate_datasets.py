"""
generate_datasets.py
Generates realistic-FEELING SYNTHETIC datasets for the Smart Water-Borne
Disease Alert Dashboard: 4 yearly disease-case files (with Cases + Deaths),
1 rainfall/humidity file, 1 state profile file, and 1 hospital directory file.

These are demo/portfolio datasets — plausible seasonal patterns and
case-fatality ratios, NOT real government surveillance records.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)

STATES = ["Assam", "Meghalaya", "Tripura", "Manipur",
          "Mizoram", "Nagaland", "Arunachal Pradesh", "Sikkim"]

STATE_CAPITALS = {
    "Assam": {"capital": "Guwahati", "lat": 26.1445, "lon": 91.7362},
    "Meghalaya": {"capital": "Shillong", "lat": 25.5788, "lon": 91.8933},
    "Tripura": {"capital": "Agartala", "lat": 23.8315, "lon": 91.2868},
    "Manipur": {"capital": "Imphal", "lat": 24.8170, "lon": 93.9368},
    "Mizoram": {"capital": "Aizawl", "lat": 23.7271, "lon": 92.7176},
    "Nagaland": {"capital": "Kohima", "lat": 25.6751, "lon": 94.1086},
    "Arunachal Pradesh": {"capital": "Itanagar", "lat": 27.0844, "lon": 93.6053},
    "Sikkim": {"capital": "Gangtok", "lat": 27.3389, "lon": 88.6065},
}

RAIN_MULT = {
    "Assam": 1.20, "Meghalaya": 1.45, "Tripura": 1.00, "Manipur": 0.95,
    "Mizoram": 1.05, "Nagaland": 0.95, "Arunachal Pradesh": 1.15, "Sikkim": 0.85
}

STATE_PROFILE = {
    "Assam":              {"population_lakhs": 356, "sanitation_index": 62, "water_quality_index": 60, "literacy_pct": 72.2},
    "Meghalaya":          {"population_lakhs": 33,  "sanitation_index": 58, "water_quality_index": 55, "literacy_pct": 74.4},
    "Tripura":            {"population_lakhs": 41,  "sanitation_index": 71, "water_quality_index": 68, "literacy_pct": 87.2},
    "Manipur":            {"population_lakhs": 33,  "sanitation_index": 66, "water_quality_index": 63, "literacy_pct": 76.9},
    "Mizoram":            {"population_lakhs": 12,  "sanitation_index": 78, "water_quality_index": 74, "literacy_pct": 91.3},
    "Nagaland":           {"population_lakhs": 22,  "sanitation_index": 69, "water_quality_index": 65, "literacy_pct": 79.6},
    "Arunachal Pradesh":  {"population_lakhs": 17,  "sanitation_index": 60, "water_quality_index": 58, "literacy_pct": 65.4},
    "Sikkim":             {"population_lakhs": 7,   "sanitation_index": 82, "water_quality_index": 79, "literacy_pct": 81.4},
}

# base = typical monthly cases per ~40 lakh population at "normal" weather pressure
# cfr = illustrative case-fatality ratio range (min, max) used to derive Deaths from Cases
DISEASES = {
    "Diarrhea":     {"base": 180, "sanitation_sensitivity": 1.6, "cfr": (0.001, 0.004)},
    "Cholera":      {"base": 22,  "sanitation_sensitivity": 2.2, "cfr": (0.01, 0.03)},
    "Typhoid":      {"base": 55,  "sanitation_sensitivity": 1.8, "cfr": (0.004, 0.012)},
    "Hepatitis A":  {"base": 30,  "sanitation_sensitivity": 1.4, "cfr": (0.001, 0.005)},
    "Dysentery":    {"base": 65,  "sanitation_sensitivity": 1.7, "cfr": (0.003, 0.008)},
}

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
YEARS = [2022, 2023, 2024, 2025]

SEASONAL_RAIN = {
    "Jan": 18, "Feb": 28, "Mar": 55, "Apr": 110, "May": 220,
    "Jun": 340, "Jul": 380, "Aug": 330, "Sep": 260, "Oct": 130,
    "Nov": 40, "Dec": 15
}
SEASONAL_HUMIDITY = {
    "Jan": 62, "Feb": 60, "Mar": 63, "Apr": 72, "May": 80,
    "Jun": 90, "Jul": 92, "Aug": 91, "Sep": 87, "Oct": 78,
    "Nov": 68, "Dec": 60
}

SPIKE_PROBABILITY = 0.07          # ~7% of state-disease-months see a localized outbreak spike
SPIKE_MULTIPLIER_RANGE = (2.2, 4.5)

DISTRICTS = {
    "Assam": ["Kamrup (Guwahati)", "Dibrugarh", "Jorhat", "Silchar (Cachar)", "Nagaon"],
    "Meghalaya": ["East Khasi Hills (Shillong)", "West Garo Hills (Tura)", "Jaintia Hills"],
    "Tripura": ["West Tripura (Agartala)", "South Tripura", "Gomati"],
    "Manipur": ["Imphal West", "Imphal East", "Churachandpur"],
    "Mizoram": ["Aizawl", "Lunglei", "Champhai"],
    "Nagaland": ["Kohima", "Dimapur", "Mokokchung"],
    "Arunachal Pradesh": ["Papum Pare (Itanagar)", "West Kameng", "Lower Subansiri"],
    "Sikkim": ["East Sikkim (Gangtok)", "South Sikkim", "West Sikkim"],
}

HOSPITAL_PREFIXES = ["District Civil Hospital", "Community Health Centre", "Regional Medical Centre",
                     "Sacred Heart Hospital", "St. Mary's Hospital", "Government General Hospital"]


def generate_weather():
    rows = []
    for year in YEARS:
        year_drift = rng.normal(0, 0.03)
        for state in STATES:
            mult = RAIN_MULT[state]
            for month in MONTHS:
                base_rain = SEASONAL_RAIN[month] * mult
                rainfall = max(0, rng.normal(base_rain, base_rain * 0.18 + 5) * (1 + year_drift))
                base_hum = SEASONAL_HUMIDITY[month]
                humidity = float(np.clip(rng.normal(base_hum, 4), 30, 99))
                rows.append({
                    "Year": year, "Month": month, "State": state,
                    "Rainfall_mm": round(rainfall, 1),
                    "Humidity_%": round(humidity, 1)
                })
    return pd.DataFrame(rows)


def generate_disease_year(weather_df_year):
    rows = []
    wmap = weather_df_year.set_index(["State", "Month"])
    for state in STATES:
        sanitation = STATE_PROFILE[state]["sanitation_index"]
        pop_factor = STATE_PROFILE[state]["population_lakhs"] / 40.0
        for disease, props in DISEASES.items():
            prev_rain, prev_hum = None, None
            for month in MONTHS:
                rainfall = wmap.loc[(state, month), "Rainfall_mm"]
                humidity = wmap.loc[(state, month), "Humidity_%"]
                lag_rain = prev_rain if prev_rain is not None else rainfall
                lag_hum = prev_hum if prev_hum is not None else humidity

                sanitation_gap = max(0, (85 - sanitation) / 85)
                weather_pressure = (lag_rain / 400) * 0.6 + (lag_hum / 100) * 0.4

                expected = (
                    props["base"] * pop_factor
                    * (0.5 + weather_pressure * 1.3)
                    * (1 + sanitation_gap * props["sanitation_sensitivity"])
                )

                is_spike = rng.random() < SPIKE_PROBABILITY
                if is_spike:
                    expected *= rng.uniform(*SPIKE_MULTIPLIER_RANGE)

                expected = max(1, expected)
                cases = int(rng.poisson(expected))

                cfr = rng.uniform(*props["cfr"])
                deaths = int(rng.binomial(cases, min(cfr * 1.5, 0.5))) if is_spike else int(rng.binomial(cases, cfr))

                rows.append({
                    "Month": month, "State": state, "Disease": disease,
                    "Cases": cases, "Deaths": deaths, "Outbreak_Spike": is_spike
                })
                prev_rain, prev_hum = rainfall, humidity
    return pd.DataFrame(rows)


def generate_state_profile():
    rows = []
    for state, props in STATE_PROFILE.items():
        rows.append({
            "State": state,
            "Capital": STATE_CAPITALS[state]["capital"],
            "Latitude": STATE_CAPITALS[state]["lat"],
            "Longitude": STATE_CAPITALS[state]["lon"],
            "Population_Lakhs": props["population_lakhs"],
            "Sanitation_Index": props["sanitation_index"],
            "Water_Quality_Index": props["water_quality_index"],
            "Literacy_Rate_%": props["literacy_pct"],
        })
    return pd.DataFrame(rows)


def generate_hospitals():
    rows = []
    hid = 1000
    for state, districts in DISTRICTS.items():
        cap = STATE_CAPITALS[state]
        for district in districts:
            n_hospitals = rng.integers(2, 4)
            for _ in range(n_hospitals):
                hid += 1
                name_prefix = rng.choice(HOSPITAL_PREFIXES)
                district_short = district.split(" (")[0]
                name = f"{name_prefix}, {district_short}"
                lat = cap["lat"] + rng.normal(0, 0.35)
                lon = cap["lon"] + rng.normal(0, 0.35)
                h_type = rng.choice(["Government", "Private"], p=[0.65, 0.35])
                total_beds = int(rng.integers(20, 250))
                occupied_pct = rng.uniform(0.3, 0.95)
                available_beds = max(0, int(total_beds * (1 - occupied_pct)))
                doctors = max(2, int(total_beds / rng.uniform(8, 15)))
                medicine_status = rng.choice(
                    ["Available", "Partially Available", "Low Stock"], p=[0.55, 0.30, 0.15]
                )
                ors_iv_available = rng.choice([True, False], p=[0.8, 0.2])
                slug = name.lower().replace(" ", "").replace(",", "").replace("'", "")[:20]
                rows.append({
                    "Hospital_ID": f"H{hid}",
                    "Hospital_Name": name,
                    "State": state,
                    "District": district_short,
                    "Type": h_type,
                    "Latitude": round(lat, 4),
                    "Longitude": round(lon, 4),
                    "Total_Beds": total_beds,
                    "Available_Beds": available_beds,
                    "Doctors_On_Staff": doctors,
                    "Medicine_Stock_Status": medicine_status,
                    "ORS_IV_Fluids_Available": ors_iv_available,
                    "Contact_Phone": f"+91-{rng.integers(70000,99999)}{rng.integers(10000,99999)}",
                    "Contact_Email": f"contact.{slug}@demohospitals.example",
                })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    weather_df = generate_weather()
    weather_df.to_excel("data/NE_RainfallHumidity.xlsx", index=False)
    print("Weather rows:", len(weather_df))

    for year in YEARS:
        year_weather = weather_df[weather_df["Year"] == year]
        disease_df = generate_disease_year(year_weather)
        fname = f"data/NE_WaterBorne_{year}_Synthetic.xlsx"
        disease_df.to_excel(fname, index=False)
        n_spikes = int(disease_df["Outbreak_Spike"].sum())
        print(fname, "rows:", len(disease_df), "| outbreak spikes:", n_spikes,
              "| total deaths:", int(disease_df["Deaths"].sum()))

    profile_df = generate_state_profile()
    profile_df.to_excel("data/NE_StateProfile.xlsx", index=False)
    print("State profile rows:", len(profile_df))

    hospitals_df = generate_hospitals()
    hospitals_df.to_excel("data/NE_Hospitals.xlsx", index=False)
    print("Hospital rows:", len(hospitals_df))

    print("\nSample disease data (with Deaths):")
    print(disease_df.head(6))
    print("\nSample hospital data:")
    print(hospitals_df.head(4))
