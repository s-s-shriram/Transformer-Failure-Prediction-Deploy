import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="GRIDTITAN - Transformer Risk Intelligence",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ GRIDTITAN - Transformer Risk Intelligence System")

st.write(
    "Data-centric AI prototype for predicting transformer failure risk, "
    "identifying probable root causes, and generating prioritized maintenance decisions."
)

# ==============================
# TRANSFORMER TYPE SELECTION
# ==============================
transformer_type = st.selectbox(
    "Select Transformer Type",
    [
        "Distribution Transformer",
        "Power Transformer",
        "Pole-Mounted Transformer"
    ],
    index=0
)

st.info(
    f"Current focus: **{transformer_type}**. "
    "For PS17, the primary focus is distribution transformers used in power distribution networks."
)

uploaded_file = st.file_uploader("Upload transformer dataset CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("📌 Dataset Preview")
    st.dataframe(data.head())

    # ==============================
    # REQUIRED COLUMNS
    # ==============================
    required_cols = [
        "transformer_id",
        "load",
        "temperature",
        "voltage",
        "current",
        "power",
        "failure"
    ]

    missing_cols = [col for col in required_cols if col not in data.columns]

    if missing_cols:
        st.error(f"CSV is missing required columns: {missing_cols}")
        st.stop()

    # ==============================
    # MISSING VALUE SUMMARY
    # ==============================
    st.subheader("🔍 Missing Value Summary")
    st.write(data.isnull().sum())

    # ==============================
    # DATA-CENTRIC MISSING VALUE HANDLING
    # ==============================
    numeric_cols = ["load", "temperature", "voltage", "current", "power"]

    imputer = KNNImputer(n_neighbors=5)
    data[numeric_cols] = imputer.fit_transform(data[numeric_cols])

    # ==============================
    # FEATURE ENGINEERING
    # ==============================
    data["thermal_stress"] = data["load"] * data["temperature"]
    data["overload"] = (data["load"] > 80).astype(int)
    data["load_ratio"] = data["load"] / 100

    # Voltage deviation from normal reference voltage
    data["voltage_deviation"] = abs(data["voltage"] - 230)

    # Current stress indicator
    data["current_stress"] = data["current"] / (data["voltage"] + 1)

    # Load fluctuation per transformer
    data = data.sort_values(by=["transformer_id"])
    data["load_fluctuation"] = (
        data.groupby("transformer_id")["load"]
        .diff()
        .fillna(0)
        .abs()
    )

    features = [
        "load",
        "temperature",
        "voltage",
        "current",
        "power",
        "thermal_stress",
        "overload",
        "load_ratio",
        "voltage_deviation",
        "current_stress",
        "load_fluctuation"
    ]

    X = data[features]
    y = data["failure"]

    # ==============================
    # CLASS IMBALANCE HANDLING
    # ==============================
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    st.subheader("⚖️ Class Balance After SMOTE")
    st.write(pd.Series(y_res).value_counts())

    # ==============================
    # TRAIN TEST SPLIT
    # ==============================
    X_train, X_test, y_train, y_test = train_test_split(
        X_res,
        y_res,
        test_size=0.2,
        random_state=42
    )

    # ==============================
    # MODEL TRAINING
    # ==============================
    model = XGBClassifier(
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    # ==============================
    # MODEL PREDICTION
    # ==============================
    y_prob = model.predict_proba(X_test)[:, 1]

    # Recall-focused threshold
    threshold = 0.35
    y_pred = (y_prob >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # ==============================
    # MODEL PERFORMANCE
    # ==============================
    st.subheader("📊 Model Performance")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy * 100:.2f}%")
    col2.metric("Recall", f"{recall * 100:.2f}%")
    col3.metric("Strategy", "Recall First")

    st.info(
        "In power systems, missing a failure is more critical than a false alarm. "
        "Therefore, this prototype prioritizes recall for early failure detection."
    )

    # ==============================
    # RISK SCORING
    # ==============================
    data["risk_score"] = model.predict_proba(X)[:, 1]

    def classify_risk(score):
        if score >= 0.70:
            return "High"
        elif score >= 0.40:
            return "Medium"
        else:
            return "Low"

    data["risk_level"] = data["risk_score"].apply(classify_risk)

    # ==============================
    # ROOT CAUSE IDENTIFICATION
    # ==============================
    current_stress_threshold = data["current_stress"].quantile(0.75)
    load_fluctuation_threshold = data["load_fluctuation"].quantile(0.75)

    def identify_root_cause(row):
        if row["load"] > 85 and row["temperature"] > 40:
            return "Thermal Overload caused by excessive load and temperature stress"
        elif row["voltage_deviation"] > 15:
            return "Voltage Instability caused by abnormal voltage fluctuation"
        elif row["current_stress"] > current_stress_threshold:
            return "Electrical Stress caused by high current flow"
        elif row["load_fluctuation"] > load_fluctuation_threshold:
            return "Sudden Load Variation caused by irregular demand pattern"
        else:
            return "General Degradation Risk based on combined operating conditions"

    def maintenance_action(level, cause):
        if level == "High":
            return f"Immediate inspection required — probable cause: {cause}"
        elif level == "Medium":
            return f"Monitor closely — possible cause: {cause}"
        else:
            return "Stable condition — continue routine monitoring"

    data["root_cause"] = data.apply(identify_root_cause, axis=1)
    data["maintenance_action"] = data.apply(
        lambda row: maintenance_action(row["risk_level"], row["root_cause"]),
        axis=1
    )

    data["transformer_type"] = transformer_type

    # ==============================
    # PRIORITY MAINTENANCE QUEUE
    # ==============================
    ranked_data = data.sort_values(by="risk_score", ascending=False)

    st.subheader("🚨 Priority Maintenance Queue - Top 5 High-Risk Transformers")

    top_5 = ranked_data[
        [
            "transformer_id",
            "transformer_type",
            "risk_score",
            "risk_level",
            "root_cause",
            "maintenance_action"
        ]
    ].head(5)

    st.dataframe(top_5)

    # ==============================
    # RISK INTELLIGENCE METRICS
    # ==============================
    test_data = pd.DataFrame(X_test, columns=features)
    test_data["actual_failure"] = y_test.values
    test_data["risk_score"] = y_prob

    k = max(1, int(0.3 * len(test_data)))
    top_k = test_data.sort_values(by="risk_score", ascending=False).head(k)

    actual_failures = test_data[test_data["actual_failure"] == 1]
    captured = top_k[top_k["actual_failure"] == 1]

    recall_at_k = len(captured) / len(actual_failures) if len(actual_failures) > 0 else 0

    high_risk_test = test_data[test_data["risk_score"] >= 0.70]
    false_alarms = high_risk_test[high_risk_test["actual_failure"] == 0]
    false_alarm_rate = len(false_alarms) / len(high_risk_test) if len(high_risk_test) > 0 else 0

    lead_time = 4 + (recall_at_k * 2)

    st.subheader("⚠️ Risk Intelligence Metrics")

    c1, c2, c3 = st.columns(3)
    c1.metric("Recall@Top-K", f"{recall_at_k:.2f}")
    c2.metric("False Alarm Rate", f"{false_alarm_rate:.2f}")
    c3.metric("Estimated Lead Time", f"{lead_time:.1f} weeks")

    # ==============================
    # DOWNLOAD REPORT
    # ==============================
    csv = ranked_data[
        [
            "transformer_id",
            "transformer_type",
            "risk_score",
            "risk_level",
            "root_cause",
            "maintenance_action"
        ]
    ].to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Risk-Ranked Maintenance Report",
        data=csv,
        file_name="risk_ranked_transformers.csv",
        mime="text/csv"
    )

    # ==============================
    # FEATURE IMPORTANCE
    # ==============================
    st.subheader("📌 Feature Importance")

    fig, ax = plt.subplots()
    ax.barh(features, model.feature_importances_)
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")
    st.pyplot(fig)

    # ==============================
    # RISK LEVEL DISTRIBUTION
    # ==============================
    st.subheader("📌 Risk Level Distribution")

    fig2, ax2 = plt.subplots()
    data["risk_level"].value_counts().plot(kind="bar", ax=ax2)
    ax2.set_xlabel("Risk Level")
    ax2.set_ylabel("Count")
    ax2.set_title("Transformer Risk Level Count")
    st.pyplot(fig2)

    # ==============================
    # CONFUSION MATRIX
    # ==============================
    st.subheader("📌 Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig3, ax3 = plt.subplots()
    ax3.imshow(cm)
    ax3.set_title("Confusion Matrix")
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax3.text(j, i, cm[i, j], ha="center", va="center")

    st.pyplot(fig3)

else:
    st.info("Please upload your transformer dataset CSV file.")
