import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(
    page_title="COVID-19 India Analysis & Forecasting",
    layout="wide"
)

# ---------- DATA LOADING HELPERS ----------

@st.cache_data
def load_case_timeseries():
    df = pd.read_csv("data/case_time_series.csv")
    # Make sure column names match what you used in Colab
    # If needed, print df.columns in Colab and adjust here
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date")
    df["Daily Confirmed"] = pd.to_numeric(df["Daily Confirmed"], errors="coerce").fillna(0)
    return df


@st.cache_data
def load_state_wise():
    df_state = pd.read_csv("data/state_wise.csv")
    # Keep only needed columns
    # Adjust column names if they differ
    if "State/UT" in df_state.columns:
        df_state = df_state.rename(columns={"State/UT": "State"})
    return df_state


@st.cache_data
def load_testing():
    df_test = pd.read_csv("data/ICMR_Testing_Data")
    # Based on your file structure:
    # columns: 'day', 'totalSamplesTested', 'totalPositiveCases',
    #          'Source', 'positive_ratio', 'perday_positive', 'perday_tests'
    df_test["Date"] = pd.to_datetime(df_test["day"], dayfirst=True, errors="coerce")
    df_test = df_test.dropna(subset=["Date"])
    df_test = df_test.sort_values("Date")
    df_test["totalSamplesTested"] = pd.to_numeric(df_test["totalSamplesTested"], errors="coerce")
    df_test["totalPositiveCases"] = pd.to_numeric(df_test["totalPositiveCases"], errors="coerce")
    df_test["positive_ratio"] = pd.to_numeric(df_test["positive_ratio"], errors="coerce")
    return df_test


@st.cache_data
def load_beds():
    beds_df = pd.read_csv("data/HospitalBedsIndia.csv")
    if "State/UT" in beds_df.columns:
        beds_df = beds_df.rename(columns={"State/UT": "State"})
    beds_df["NumPublicBeds_HMIS"] = pd.to_numeric(beds_df["NumPublicBeds_HMIS"], errors="coerce")
    return beds_df


@st.cache_data
def prepare_health_df():
    df_state = load_state_wise()
    beds_df = load_beds()

    # Use your state-wise df that has Confirmed, Recovered, Deaths
    if "Confirmed" not in df_state.columns:
        return None

    df_states = df_state[df_state["State"] != "Total"].copy()
    df_states["Confirmed"] = pd.to_numeric(df_states["Confirmed"], errors="coerce")

    health_df = pd.merge(df_states, beds_df, on="State", how="inner")
    health_df["NumPublicBeds_HMIS"] = pd.to_numeric(health_df["NumPublicBeds_HMIS"], errors="coerce")
    health_df["CasesPerBed"] = health_df["Confirmed"] / health_df["NumPublicBeds_HMIS"]
    health_df = health_df.dropna(subset=["CasesPerBed"])
    return health_df


@st.cache_data
def train_arima(series, order=(2, 1, 2), steps=30):
    model = SARIMAX(series, order=order)
    res = model.fit(disp=False)
    forecast = res.get_forecast(steps=steps)
    pred_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()
    return pred_mean, conf_int


# ---------- LAYOUT ----------

st.title("🇮🇳 COVID-19 India Analysis & Forecasting")
st.write("Interactive dashboard for spread analysis, testing trends, healthcare capacity, and forecasting.")

section = st.sidebar.radio(
    "Select a section",
    [
        "National Trend",
        "State-wise Analysis",
        "Testing & Positivity",
        "Healthcare Capacity",
        "Forecast (ARIMA)"
    ]
)

# ---------- SECTION: NATIONAL TREND ----------

if section == "National Trend":
    df = load_case_timeseries()

    st.subheader("Daily Confirmed COVID-19 Cases in India")

    col1, col2 = st.columns([3, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df["Date"], df["Daily Confirmed"])
        ax.set_title("Daily Confirmed Cases")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cases")
        ax.grid(True)
        st.pyplot(fig)

    with col2:
        st.markdown("**Key Points**")
        st.write(f"- Records from: **{df['Date'].min().date()}** to **{df['Date'].max().date()}**")
        st.write("- Peaks correspond to major COVID-19 waves in India.")
        st.write("- Use this section to understand overall national trend.")


# ---------- SECTION: STATE-WISE ANALYSIS ----------

elif section == "State-wise Analysis":
    df_state = load_state_wise()
    df_states = df_state[df_state["State"] != "Total"].copy()
    df_states["Confirmed"] = pd.to_numeric(df_states["Confirmed"], errors="coerce")
    df_states["Recovered"] = pd.to_numeric(df_states["Recovered"], errors="coerce")
    df_states["Deaths"] = pd.to_numeric(df_states["Deaths"], errors="coerce")

    st.subheader("State-wise COVID-19 Impact")

    # Top N slider
    top_n = st.sidebar.slider("Top N states by confirmed cases", 5, 20, 10)

    # Bar chart for top N states
    top_states = df_states.sort_values("Confirmed", ascending=False).head(top_n)

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(top_states["State"], top_states["Confirmed"])
    ax1.set_title(f"Top {top_n} States by Confirmed Cases")
    ax1.set_xlabel("State")
    ax1.set_ylabel("Total Confirmed Cases")
    ax1.tick_params(axis='x', rotation=45)
    st.pyplot(fig1)

    # Recovery vs Fatality rates
    df_states["Recovery Rate"] = (df_states["Recovered"] / df_states["Confirmed"]) * 100
    df_states["Fatality Rate"] = (df_states["Deaths"] / df_states["Confirmed"]) * 100

    st.subheader("Recovery vs Fatality Rate (Top States)")
    rates_top = df_states.sort_values("Confirmed", ascending=False).head(top_n)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(rates_top["State"], rates_top["Recovery Rate"], marker='o', label="Recovery Rate (%)")
    ax2.plot(rates_top["State"], rates_top["Fatality Rate"], marker='o', linestyle='--', label="Fatality Rate (%)")
    ax2.set_xlabel("State")
    ax2.set_ylabel("Rate (%)")
    ax2.set_title("Recovery vs Fatality Rates")
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)


# ---------- SECTION: TESTING & POSITIVITY ----------

elif section == "Testing & Positivity":
    df_test = load_testing()

    st.subheader("Testing Growth Over Time")

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df_test["Date"], df_test["totalSamplesTested"])
    ax1.set_title("Total Samples Tested Over Time")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Total Samples Tested")
    ax1.grid(True)
    st.pyplot(fig1)

    st.subheader("Positivity Ratio Trend")

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(df_test["Date"], df_test["positive_ratio"], color="red")
    ax2.set_title("Positivity Ratio Over Time")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Positivity Ratio (%)")
    ax2.grid(True)
    st.pyplot(fig2)

    st.markdown("**Interpretation:**")
    st.write("- Rising testing volume indicates scaling of health response.")
    st.write("- Higher positivity spikes show major waves and periods of under-testing.")


# ---------- SECTION: HEALTHCARE CAPACITY ----------

elif section == "Healthcare Capacity":
    health_df = prepare_health_df()

    if health_df is None:
        st.error("Healthcare capacity data could not be prepared. Please verify state_wise and HospitalBedsIndia files.")
    else:
        st.subheader("COVID-19 Burden vs Public Hospital Beds")

        # Sort by highest case burden per bed
        stress_top10 = health_df.sort_values("CasesPerBed", ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(stress_top10["State"], stress_top10["CasesPerBed"], color="orange")
        ax.set_title("Top 10 States by Cases per Public Hospital Bed")
        ax.set_xlabel("State")
        ax.set_ylabel("Cases per Bed (Higher = More Stress)")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True)
        st.pyplot(fig)

        st.markdown("**Interpretation:**")
        st.write("- Higher cases-per-bed indicate more strain on healthcare capacity.")
        st.write("- Can be used to prioritize resource allocation.")


# ---------- SECTION: FORECAST (ARIMA) ----------

elif section == "Forecast (ARIMA)":
    df = load_case_timeseries()
    ts = df.set_index("Date")["Daily Confirmed"].asfreq("D").fillna(0)

    st.subheader("ARIMA-based Forecast of Daily Cases")

    steps = st.sidebar.slider("Days to forecast", min_value=7, max_value=60, value=30, step=1)

    with st.spinner("Training ARIMA model..."):
        pred_mean, conf_int = train_arima(ts, steps=steps)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ts.index, ts.values, label="Historical")
    ax.plot(pred_mean.index, pred_mean.values, label="Forecast", linestyle='--')

    # Confidence interval
    ax.fill_between(
        conf_int.index,
        conf_int.iloc[:, 0],
        conf_int.iloc[:, 1],
        alpha=0.3,
        label="Confidence Interval"
    )

    ax.set_title(f"Forecast of Daily Confirmed Cases (Next {steps} Days)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Confirmed Cases")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.markdown("**Notes:**")
    st.write("- ARIMA model is statistical and based on historical trends only.")
    st.write("- Useful for short-term planning and understanding possible trajectories.")
