import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.express as px
import shap
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Dengue Outbreak Predictor",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# UNIT CONVERSION HELPERS
# ------------------------------------------------------------
# The training data (Visual Crossing Weather) was pulled in US
# units: temp in °F, precip in inches, wind in mph. The UI below
# stays in metric (Celsius / mm / km-h) for a Bangladeshi
# audience, and we convert right before scoring so the model
# sees the same units it was trained on.
# ============================================================
def c_to_f(c):
    return c * 9 / 5 + 32

def mm_to_in(mm):
    return mm / 25.4

def kmh_to_mph(kmh):
    return kmh / 1.60934

# ============================================================
# LOAD MODELS (Cached for performance)
# ============================================================
@st.cache_resource
def load_models():
    try:
        rf_model = joblib.load('models/rf_classifier_v2.pkl')
        # NOTE: scaler_v2.pkl is loaded for completeness/inspection but is
        # intentionally NOT applied to the RF's input — see the comment above
        # `rf_input = input_data` further down for why.
        scaler = joblib.load('models/scaler_v2.pkl')
        # compile=False: skips deserializing the optimizer/metrics,
        # which is what breaks load_model() across Keras 2 vs Keras 3
        # save formats. We only need forward inference, not training
        # state, so this is safe and makes loading version-agnostic.
        lstm_model = tf.keras.models.load_model('models/lstm_regressor.h5', compile=False)
        features = joblib.load('models/feature_cols_v2.pkl')
        threshold = joblib.load('models/rf_threshold.pkl')

        # The LSTM was trained on the ORIGINAL 20-feature set and was never
        # retrained alongside the RF classifier upgrade (rolling features
        # were added only for the RF fix). It still needs its own 20-column
        # scaler + feature list, separate from the RF's new 23-column ones,
        # or its input shape won't match what it was trained on.
        lstm_scaler = joblib.load('models/scaler.pkl')
        lstm_features = joblib.load('models/feature_columns.pkl')

        return rf_model, scaler, lstm_model, features, threshold, lstm_scaler, lstm_features
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.exception(e)
        st.stop()

rf_model, scaler, lstm_model, feature_cols, RF_THRESHOLD, lstm_scaler, lstm_feature_cols = load_models()

# ============================================================
# MODEL DIAGNOSTIC — confirms which rf_classifier.pkl is actually
# loaded in memory. @st.cache_resource keeps models cached for the
# life of the server process, so swapping the .pkl file on disk has
# NO effect until the Streamlit process is fully stopped (Ctrl+C)
# and restarted — a browser refresh alone does not reload it.
# ============================================================
import hashlib
with open('models/rf_classifier_v2.pkl', 'rb') as _f:
    _file_hash = hashlib.md5(_f.read()).hexdigest()[:8]

st.sidebar.markdown("---")
with st.sidebar.expander("🔧 Loaded Model Info (debug)"):
    st.caption(f"class_weight: **{rf_model.class_weight}**")
    st.caption(f"n_estimators: {rf_model.n_estimators}")
    st.caption(f"n_features: {rf_model.n_features_in_}")
    st.caption(f"decision threshold: **{RF_THRESHOLD}**")
    st.caption(f"file hash: {_file_hash}")
    if rf_model.class_weight != 'balanced':
        st.warning("⚠️ This is the ORIGINAL classifier, not the retrained one. If you replaced rf_classifier_v2.pkl, fully stop (Ctrl+C) and restart `streamlit run` — a browser refresh alone won't reload a cached model.")

@st.cache_resource
def get_shap_explainer(_model):
    return shap.TreeExplainer(_model)

shap_explainer = get_shap_explainer(rf_model)

# ============================================================
# LOAD + REBUILD HISTORICAL DATA FOR ANALYTICS
# ------------------------------------------------------------
# There is no pre-merged 'final_dengue_climate_all_divisions.csv'
# on disk — only the two raw files. This rebuilds the same join
# your notebook does (cases by division/week + weekly climate
# aggregates), mirroring the cleaning logic from the training
# notebook so the numbers are consistent with the trained model.
# ============================================================
DIVISION_MAP = {
    'Chattogram Division (Out of CC)': 'Chittagong',
    'Chittagong Division (Out of CC)': 'Chittagong',
    'Barishal Division (Out of CC)': 'Barishal',
    'Khulna Division (Out of CC)': 'Khulna',
    'Mymensingh Division (Out of CC)': 'Mymensingh',
    'Rajshahi Division (Out of CC)': 'Rajshahi',
    'Rangpur Division (Out of CC)': 'Rangpur',
    'Sylhet Division (Out of CC)': 'Sylhet',
    'Dhaka Division (Out of CC)': 'Dhaka',
    'DNCC': 'Dhaka',
    'DSCC': 'Dhaka',
}
CLIMATE_NAME_MAP = {
    'Dhaka, Bangladesh': 'Dhaka',
    'Chittagong, Bangladesh': 'Chittagong',
    'Barishal, Bangladesh': 'Barishal',
    'Khulna, Bangladesh': 'Khulna',
    'Mymensingh, Bangladesh': 'Mymensingh',
    'Rajshahi, Bangladesh': 'Rajshahi',
    'Rangpur, Bangladesh': 'Rangpur',
    'Sylhet, Bangladesh': 'Sylhet',
}

@st.cache_data
def load_historical_data():
    try:
        dengue = pd.read_csv('data/dengue_raw.csv')
        dengue.columns = ['year', 'week', 'division', 'cases']
        dengue['cases'] = pd.to_numeric(
            dengue['cases'].replace('null', 0), errors='coerce'
        ).fillna(0).astype(int)
        dengue['division'] = dengue['division'].str.strip().map(DIVISION_MAP)
        dengue = dengue.dropna(subset=['division'])
        dengue = dengue.groupby(['year', 'week', 'division'], as_index=False)['cases'].sum()

        climate = pd.read_csv('data/climate_daily.csv').dropna(subset=['name'])
        climate['datetime'] = pd.to_datetime(climate['datetime'])
        climate['year'] = climate['datetime'].dt.year
        climate['week'] = climate['datetime'].dt.isocalendar().week.astype(int)
        climate['division'] = climate['name'].map(CLIMATE_NAME_MAP)
        climate = climate.dropna(subset=['division'])

        weekly_climate = climate.groupby(['division', 'year', 'week'], as_index=False).agg(
            temp=('temp', 'mean'),
            humidity=('humidity', 'mean'),
            precip=('precip', 'sum'),
            windspeed=('windspeed', 'mean'),
            uvindex=('uvindex', 'mean'),
        )

        merged = pd.merge(dengue, weekly_climate, on=['division', 'year', 'week'], how='inner')

        # Convert to metric purely for display - doesn't touch the model.
        merged['temp'] = (merged['temp'] - 32) * 5 / 9
        merged['precip'] = merged['precip'] * 25.4

        threshold = merged['cases'].quantile(0.75)
        merged['outbreak'] = (merged['cases'] > threshold).astype(int)

        return merged.sort_values(['division', 'year', 'week']).reset_index(drop=True)
    except Exception as e:
        st.warning(f"Could not rebuild historical analytics data: {e}")
        return None

historical_df = load_historical_data()

# ============================================================
# REAL EXAMPLE WEEKS — for the "Load real example" picker.
# ------------------------------------------------------------
# The manual sliders + Stable/Building/Cooling trend toggle can only
# express a strictly linear lag pattern. Real outbreak-predictive weeks
# have non-monotonic lag trajectories (e.g. humidity rising then
# falling) that the simplified trend toggle structurally cannot
# reproduce. These 5 rows are real test-set weeks (with real model
# probabilities + ground truth) pulled directly from the training
# notebook, so selecting one here reproduces the model's actual,
# validated behavior rather than an invented scenario.
# ============================================================
@st.cache_data
def load_example_weeks():
    try:
        with open('data/example_weeks.json', 'r') as f:
            return json.load(f)
    except Exception:
        return []

example_weeks = load_example_weeks()

# ============================================================
# SIDEBAR - INPUT SECTION
# ============================================================
st.sidebar.title("🦟 Dengue Predictor")
st.sidebar.markdown("---")

divisions = ['Barishal', 'Chittagong', 'Dhaka', 'Khulna', 'Mymensingh', 'Rajshahi', 'Rangpur', 'Sylhet']
division = st.sidebar.selectbox("📍 Select Division", divisions)
st.sidebar.caption("Used for the Analytics tab. The current model scores climate alone, not division - see note in the Predict tab.")

st.sidebar.markdown("---")
st.sidebar.subheader("📂 Load a Real Example Week")
st.sidebar.caption(
    "Manual sliders below use a simplified linear trend that can't fully "
    "reproduce the model's real lag patterns. These are unmodified weeks "
    "from the test set with the model's actual validated output — the "
    "most reliable way to see real risk separation."
)

if example_weeks:
    example_labels = ["— None, use manual inputs below —"] + [ex["label"] for ex in example_weeks]
    selected_label = st.sidebar.selectbox("Example week", example_labels)
    selected_example = next((ex for ex in example_weeks if ex["label"] == selected_label), None)

    if selected_example:
        st.sidebar.success(
            f"Loaded — model scored this real week at "
            f"**{selected_example['model_prob']*100:.1f}%** "
            f"(actual outcome: {'Outbreak' if selected_example['true_label'] == 1 else 'No outbreak'})"
        )
else:
    selected_example = None
    st.sidebar.caption("⚠️ data/example_weeks.json not found — manual inputs only.")

st.sidebar.markdown("---")
st.sidebar.subheader("📊 This Week's Climate (avg/total)")

# When an example is loaded, its values become the widget defaults.
# Streamlit number_input only applies `value` on first render per key,
# so we force a fresh key per example to make the defaults actually update.
_ex_key = selected_example["id"] if selected_example else "manual"

col1, col2 = st.sidebar.columns(2)
with col1:
    temp = st.number_input(
        "🌡️ Avg Temp (°C)", 15.0, 40.0,
        selected_example["temp_c"] if selected_example else 28.5,
        step=0.5, key=f"temp_{_ex_key}"
    )
    tempmax = st.number_input(
        "🌡️ Avg High (°C)", 18.0, 45.0,
        selected_example["tempmax_c"] if selected_example else temp + 5,
        step=0.5, key=f"tempmax_{_ex_key}"
    )
    humidity = st.number_input(
        "💧 Humidity (%)", 30.0, 100.0,
        selected_example["humidity"] if selected_example else 75.0,
        step=1.0, key=f"humidity_{_ex_key}"
    )
    precip = st.number_input(
        "🌧️ Total Rain (mm)", 0.0, 400.0,
        selected_example["precip_mm"] if selected_example else 50.0,
        step=5.0, key=f"precip_{_ex_key}"
    )
with col2:
    tempmin = st.number_input(
        "🌡️ Avg Low (°C)", 5.0, 35.0,
        selected_example["tempmin_c"] if selected_example else temp - 5,
        step=0.5, key=f"tempmin_{_ex_key}"
    )
    windspeed = st.number_input(
        "💨 Wind Speed (km/h)", 0.0, 60.0,
        selected_example["windspeed_kmh"] if selected_example else 15.0,
        step=0.5, key=f"windspeed_{_ex_key}"
    )
    cloudcover = st.number_input(
        "☁️ Cloud Cover (%)", 0.0, 100.0,
        selected_example["cloudcover"] if selected_example else 50.0,
        step=5.0, key=f"cloudcover_{_ex_key}"
    )
    uvindex = st.number_input(
        "☀️ UV Index", 0, 12,
        int(round(selected_example["uvindex"])) if selected_example else 7,
        key=f"uvindex_{_ex_key}"
    )

week = st.sidebar.number_input("📅 Week Number", 1, 52, 30)

st.sidebar.markdown("---")
st.sidebar.subheader("📅 Past 4-Week Trend")
if selected_example:
    st.sidebar.caption("Ignored — using the real example's actual lag values instead (see below).")
trend = st.sidebar.selectbox(
    "How did conditions trend into this week?",
    [
        "Stable (similar to this week)",
        "Building up (cooler/drier 4 weeks ago)",
        "Cooling down (hotter/wetter 4 weeks ago)",
    ],
    disabled=bool(selected_example),
)

if selected_example:
    # Real example selected — use its exact (non-monotonic) lag values
    # directly. This is the whole point of the picker: these reproduce
    # the model's actual validated behavior, not an approximation.
    temp_lag1, temp_lag2, temp_lag3, temp_lag4 = selected_example["temp_lag_c"]
    humidity_lag1, humidity_lag2, humidity_lag3, humidity_lag4 = selected_example["humidity_lag"]
    precip_lag1, precip_lag2, precip_lag3, precip_lag4 = selected_example["precip_lag_mm"]
    trend = "Loaded from real example (overrides trend toggle above)"
else:
    if trend.startswith("Stable"):
        direction = 0
    elif trend.startswith("Building"):
        direction = -1   # conditions 4 weeks ago were milder than today
    else:
        direction = 1    # conditions 4 weeks ago were more extreme than today

    temp_step = 0.5 * direction
    humidity_step = 2.0 * direction
    precip_step = 10.0 * direction

    temp_lag1 = float(np.clip(temp + temp_step * 1, 5, 45))
    temp_lag2 = float(np.clip(temp + temp_step * 2, 5, 45))
    temp_lag3 = float(np.clip(temp + temp_step * 3, 5, 45))
    temp_lag4 = float(np.clip(temp + temp_step * 4, 5, 45))

    humidity_lag1 = float(np.clip(humidity + humidity_step * 1, 30, 100))
    humidity_lag2 = float(np.clip(humidity + humidity_step * 2, 30, 100))
    humidity_lag3 = float(np.clip(humidity + humidity_step * 3, 30, 100))
    humidity_lag4 = float(np.clip(humidity + humidity_step * 4, 30, 100))

    precip_lag1 = max(0.0, precip + precip_step * 1)
    precip_lag2 = max(0.0, precip + precip_step * 2)
    precip_lag3 = max(0.0, precip + precip_step * 3)
    precip_lag4 = max(0.0, precip + precip_step * 4)

with st.sidebar.expander("🔍 View computed lag values"):
    if selected_example:
        st.caption("Real values from the loaded example week (not derived from the trend toggle):")
    else:
        st.caption("Auto-updates from the trend above and this week's values:")
    st.write(f"**Temp:** {temp_lag1:.1f}, {temp_lag2:.1f}, {temp_lag3:.1f}, {temp_lag4:.1f} °C (1–4 wks ago)")
    st.write(f"**Humidity:** {humidity_lag1:.0f}, {humidity_lag2:.0f}, {humidity_lag3:.0f}, {humidity_lag4:.0f} % (1–4 wks ago)")
    st.write(f"**Rain:** {precip_lag1:.0f}, {precip_lag2:.0f}, {precip_lag3:.0f}, {precip_lag4:.0f} mm (1–4 wks ago)")

# ============================================================
# PREPARE INPUT DATA (convert metric UI -> US units the model expects)
# ============================================================
input_data = pd.DataFrame([{
    'temp': c_to_f(temp), 'tempmax': c_to_f(tempmax), 'tempmin': c_to_f(tempmin),
    'humidity': humidity, 'precip': mm_to_in(precip), 'windspeed': kmh_to_mph(windspeed),
    'cloudcover': cloudcover, 'uvindex': uvindex,
    'temp_lag1': c_to_f(temp_lag1), 'temp_lag2': c_to_f(temp_lag2),
    'temp_lag3': c_to_f(temp_lag3), 'temp_lag4': c_to_f(temp_lag4),
    'humidity_lag1': humidity_lag1, 'humidity_lag2': humidity_lag2,
    'humidity_lag3': humidity_lag3, 'humidity_lag4': humidity_lag4,
    'precip_lag1': mm_to_in(precip_lag1), 'precip_lag2': mm_to_in(precip_lag2),
    'precip_lag3': mm_to_in(precip_lag3), 'precip_lag4': mm_to_in(precip_lag4),
}])

# LSTM input: built from the original 20 columns only, scaled with the
# ORIGINAL scaler.pkl. The LSTM was never retrained with the rolling
# features added for the RF fix, so it must keep seeing exactly the
# 20-column shape it was trained on.
lstm_input_data = input_data[lstm_feature_cols]
lstm_scaled_input = lstm_scaler.transform(lstm_input_data)

# Rolling/cumulative features (v2 model) — same formula used in training:
# mean of the four lag columns for each climate variable. Must be computed
# from the *converted* (°F / inches) lag columns above, since that's what
# the model was trained on.
input_data['precip_roll4'] = input_data[['precip_lag1', 'precip_lag2', 'precip_lag3', 'precip_lag4']].mean(axis=1)
input_data['humidity_roll4'] = input_data[['humidity_lag1', 'humidity_lag2', 'humidity_lag3', 'humidity_lag4']].mean(axis=1)
input_data['temp_roll4'] = input_data[['temp_lag1', 'temp_lag2', 'temp_lag3', 'temp_lag4']].mean(axis=1)

# Ensure correct column order (must match the order feature_cols_v2 was trained with)
input_data = input_data[feature_cols]

# IMPORTANT: rf_clf_v2 was trained directly on raw (unscaled) feature values —
# see notebook cell 68, `rf_clf_v2.fit(X_train_v2, y_train_clf)` — X_train_v2
# is never passed through scaler_v2.fit_transform() before fitting. scaler_v2
# (cell 73) is fit afterward and was apparently intended for deployment, but
# it was never the data the model actually learned from. Verified directly:
# scoring example_weeks.json rows through scaler_v2.transform() before predict_proba
# collapses every example into a ~0.21-0.25 band regardless of true risk, while
# scoring the same raw rows reproduces the notebook's recorded probabilities
# almost exactly (e.g. high_2: expected 0.832, raw-scored 0.832, scaled-scored 0.222).
# So the RF gets the raw input_data directly — no scaling.
rf_input = input_data

# ============================================================
# MAIN CONTENT - 3 TABS
# ============================================================
st.title("🦟 Dengue Outbreak Prediction Dashboard")
st.markdown("*AI-powered Early Warning System for Bangladesh*")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Analytics", "🔍 Explain"])

# ============================================================
# TAB 1: PREDICT
# ============================================================
with tab1:
    st.header("🔮 Outbreak Risk Prediction")

    # ── Model context banner ────────────────────────────────
    st.info(
        "**How this works:** A Random Forest classifier (ROC-AUC 0.95 on test data) scores "
        "weekly climate conditions against the historical outbreak threshold (75th percentile "
        "of regional case counts). The risk probability below is the primary validated output. "
        "An LSTM case-count estimate is also shown as an **experimental** secondary indicator — "
        "trained on a single year of data, it should be treated as directional, not precise."
    )

    st.markdown("---")

    col1, col2, col3 = st.columns([1, 1.2, 1])

    with col1:
        st.markdown("### 📋 Current Inputs")
        st.write(f"**Division:** {division}")
        st.write(f"**Week:** {week}")
        st.write(f"**Avg Temperature:** {temp}°C")
        st.write(f"**Humidity:** {humidity}%")
        st.write(f"**Total Rain:** {precip} mm")
        st.write(f"**Wind Speed:** {windspeed} km/h")
        st.write(f"**UV Index:** {uvindex}")
        st.write(f"**Cloud Cover:** {cloudcover}%")

    with col2:
        st.markdown("### 🚨 Risk Assessment")

        if st.button("▶ Run Prediction", type="primary", use_container_width=True):
            with st.spinner("Running models..."):
                outbreak_prob = rf_model.predict_proba(rf_input)[0][1]

                lstm_input = lstm_scaled_input.reshape(1, 1, -1)
                predicted_cases = max(0, lstm_model.predict(lstm_input, verbose=0)[0][0])

                if outbreak_prob >= RF_THRESHOLD:
                    risk_level = "🔴 HIGH RISK"
                    risk_color = "error"
                    alert_msg = "⚠️ High outbreak probability — escalate surveillance now."
                elif outbreak_prob >= RF_THRESHOLD * 0.5:
                    risk_level = "🟡 MEDIUM RISK"
                    risk_color = "warning"
                    alert_msg = "⚠️ Elevated risk — monitor closely and prepare resources."
                else:
                    risk_level = "🟢 LOW RISK"
                    risk_color = "success"
                    alert_msg = "✅ Below outbreak threshold — maintain routine surveillance."

                # ── Primary output: RF classification ───────
                st.markdown("#### Primary Output — Random Forest Classifier")
                st.metric(
                    label="Outbreak Risk Level",
                    value=risk_level,
                    delta=f"{outbreak_prob * 100:.1f}% outbreak probability"
                )

                fig, ax = plt.subplots(figsize=(7, 1.6))
                bar_color = '#e74c3c' if outbreak_prob >= RF_THRESHOLD else '#f39c12' if outbreak_prob >= RF_THRESHOLD * 0.5 else '#27ae60'
                ax.barh(['Risk'], [outbreak_prob * 100], color=bar_color, height=0.5)
                ax.set_xlim(0, 100)
                ax.set_xlabel('Outbreak Probability (%)', fontsize=9)
                ax.axvline(RF_THRESHOLD * 50, color='#f39c12', linestyle='--', linewidth=1.2, alpha=0.8, label=f'Medium ({RF_THRESHOLD * 50:.0f}%)')
                ax.axvline(RF_THRESHOLD * 100, color='#e74c3c', linestyle='--', linewidth=1.2, alpha=0.8, label=f'High ({RF_THRESHOLD * 100:.0f}%)')
                ax.legend(fontsize=8, loc='lower right')
                ax.set_title('Risk Probability (Random Forest — ROC-AUC 0.95, Recall 1.00)', fontsize=9)
                ax.tick_params(axis='y', labelsize=9)
                st.pyplot(fig)
                plt.close(fig)

                if risk_color == "error":
                    st.error(alert_msg)
                elif risk_color == "warning":
                    st.warning(alert_msg)
                else:
                    st.success(alert_msg)

                # ── Secondary output: LSTM case estimate ─────
                st.markdown("---")
                st.markdown("#### Secondary Output — LSTM Case Estimate *(Experimental)*")
                st.metric(
                    label="Estimated Weekly Cases",
                    value=f"~{int(predicted_cases):,}",
                )
                st.caption(
                    "⚠️ **Treat as directional only.** The LSTM regressor was trained on a "
                    "single year of 2025 surveillance data (394 samples), which is insufficient "
                    "for reliable case-count regression (test R² = −0.37). The value indicates "
                    "rough order-of-magnitude scale, not a precise forecast. "
                    "See the Explain tab for which climate features most influenced this result."
                )

                st.session_state['outbreak_prob'] = outbreak_prob
                st.session_state['predicted_cases'] = predicted_cases
                st.session_state['rf_input'] = rf_input

        else:
            st.info("☝️ Click **Run Prediction** to score current climate inputs.")

    with col3:
        st.markdown("### 📋 Recommended Actions")
        if 'outbreak_prob' in st.session_state:
            prob = st.session_state['outbreak_prob']
            if prob >= RF_THRESHOLD:
                st.error("**🔴 Immediate Actions:**")
                st.write("- Intensify mosquito vector control")
                st.write("- Alert divisional hospitals")
                st.write("- Launch public awareness campaigns")
                st.write("- Pre-position medical supplies")
            elif prob >= RF_THRESHOLD * 0.5:
                st.warning("**🟡 Preventive Actions:**")
                st.write("- Increase field surveillance frequency")
                st.write("- Inspect stagnant water sites")
                st.write("- Brief local health offices")
                st.write("- Review hospital capacity")
            else:
                st.success("**🟢 Routine Actions:**")
                st.write("- Continue weekly surveillance")
                st.write("- Maintain data collection")
                st.write("- Monitor upcoming weather forecasts")
                st.write("- Keep response plans current")

            st.markdown("---")
            st.markdown("**📌 Model Info**")
            st.caption(f"Classifier: Random Forest (200 trees, max_depth=8) | ROC-AUC: 0.95 | Recall: 1.00 | Precision: 0.43 | Decision threshold: {RF_THRESHOLD} | Training data: 2025 Bangladesh divisions")
            st.caption("Outbreak defined as > 75th percentile of weekly regional case distribution. Threshold tuned to maximize F1 while preserving 100% recall — the model is intentionally biased toward catching every real outbreak at the cost of some false alarms, the safer tradeoff for public health early-warning.")
        else:
            st.info("Run a prediction to see recommended actions.")

# ============================================================
# TAB 2: ANALYTICS
# ============================================================
with tab2:
    st.header("📊 Historical Analytics & Trends")

    if historical_df is not None:
        df_filtered = historical_df[historical_df['division'] == division]

        if not df_filtered.empty:
            st.subheader(f"📈 Dengue Cases Over Time - {division}")
            fig = px.line(
                df_filtered, x='week', y='cases',
                title=f'Weekly Cases in {division}',
                labels={'week': 'Week Number', 'cases': 'Number of Cases'},
                color_discrete_sequence=['#e74c3c']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("🌡️ Climate vs Cases")
            col1, col2 = st.columns(2)
            with col1:
                fig2 = px.scatter(
                    df_filtered, x='temp', y='cases',
                    title='Temperature vs Cases',
                    labels={'temp': 'Temperature (°C)', 'cases': 'Cases'},
                    color_discrete_sequence=['#3498db']
                )
                fig2.update_layout(height=350)
                st.plotly_chart(fig2, use_container_width=True)
            with col2:
                fig3 = px.scatter(
                    df_filtered, x='humidity', y='cases',
                    title='Humidity vs Cases',
                    labels={'humidity': 'Humidity (%)', 'cases': 'Cases'},
                    color_discrete_sequence=['#2ecc71']
                )
                fig3.update_layout(height=350)
                st.plotly_chart(fig3, use_container_width=True)

            st.subheader("📊 Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Cases", f"{int(df_filtered['cases'].sum()):,}")
            with col2:
                st.metric("Avg Weekly Cases", f"{df_filtered['cases'].mean():.1f}")
            with col3:
                st.metric("Max Weekly Cases", f"{int(df_filtered['cases'].max()):,}")
            with col4:
                st.metric("Outbreak Weeks", f"{len(df_filtered[df_filtered['outbreak']==1])}")
        else:
            st.info(f"No data available for {division}")
    else:
        st.warning("Historical data could not be loaded. Check that data/dengue_raw.csv and data/climate_daily.csv are present.")

# ============================================================
# TAB 3: EXPLAIN (real SHAP)
# ============================================================
with tab3:
    st.header("🔍 Explainable AI - Why This Prediction?")

    if 'rf_input' in st.session_state:
        st.write(f"**Current Risk Probability:** {st.session_state['outbreak_prob']*100:.1f}%")

        st.subheader("📊 SHAP: What Drove This Specific Prediction")

        shap_values = shap_explainer.shap_values(st.session_state['rf_input'])
        # shap >=0.45 returns shape (n_samples, n_features, n_classes);
        # older versions return a list of per-class arrays. Handle both.
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        elif shap_values.ndim == 3:
            sv = shap_values[0, :, 1]
        else:
            sv = shap_values[0]

        shap_df = pd.DataFrame({'Feature': feature_cols, 'SHAP': sv})
        shap_df = shap_df.reindex(shap_df['SHAP'].abs().sort_values(ascending=False).index).head(10)

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#e74c3c' if v > 0 else '#3498db' for v in shap_df['SHAP']]
        ax.barh(shap_df['Feature'], shap_df['SHAP'], color=colors)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('SHAP value (impact on outbreak probability)')
        ax.set_title('Top 10 Features Driving This Prediction')
        ax.invert_yaxis()
        st.pyplot(fig)
        st.caption("🔴 Red = pushes risk higher · 🔵 Blue = pushes risk lower. Bar length = size of that feature's impact on this specific prediction.")

        st.subheader("💡 What This Means")
        top_features = shap_df.head(3)['Feature'].tolist()
        explanation = f"""
        **Prediction:** {st.session_state['outbreak_prob']*100:.1f}% risk of dengue outbreak

        **Key Drivers (by SHAP impact):**
        1. **{top_features[0].replace('_', ' ').title()}**
        2. **{top_features[1].replace('_', ' ').title()}**
        3. **{top_features[2].replace('_', ' ').title()}**
        """
        st.info(explanation)

        st.subheader("🌡️ Climate Pattern Analysis")
        st.write(f"""
        - **Temperature:** {temp}°C {'(Higher than average)' if temp > 28 else '(Normal)'}
        - **Humidity:** {humidity}% {'(High - favorable for mosquito breeding)' if humidity > 75 else '(Normal)'}
        - **Precipitation:** {precip}mm {'(Increased - potential for stagnant water)' if precip > 60 else '(Normal)'}
        """)

        st.markdown("---")
        st.caption("⚠️ **Note:** This model was trained on 2025 data only. Predictions are climate-based and should be used as guidance, not definitive medical advice.")
    else:
        st.info("Please run a prediction in the 'Predict' tab first to see explanations here.")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("🦟 Built with ❤️ | Powered by Random Forest & LSTM | Data: DGHS & Visual Crossing Weather |")