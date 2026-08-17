

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Intrusion Profiler",
    page_icon="🛡️",
    layout="wide"
)

# ── Column names
COLUMNS = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
]
# NOTE: 'attack_type' and 'difficulty_level' are NOT included —
# uploaded traffic to classify won't have a known label.

CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']



@st.cache_resource
def load_models():
    rf = joblib.load("rf_classifier.pkl")
    iso_forest = joblib.load("isolation_forest.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    feature_encoders = joblib.load("feature_encoders.pkl")
    return rf, iso_forest, scaler, label_encoder, feature_encoders


# ── Hybrid prediction logic (same as the notebook) ──────────
def hybrid_predict(X_scaled, rf_model, iso_model, label_encoder, class_names):
    rf_pred = rf_model.predict(X_scaled)
    rf_proba = rf_model.predict_proba(X_scaled)
    rf_confidence = rf_proba.max(axis=1)

    anomaly_raw = iso_model.predict(X_scaled)
    is_anomaly = (anomaly_raw == -1)

    normal_idx = label_encoder.transform(['normal'])[0]

    verdicts, reasons, confidences = [], [], []
    for i in range(len(X_scaled)):
        rf_label = class_names[rf_pred[i]]
        if rf_pred[i] != normal_idx:
            verdicts.append(rf_label)
            reasons.append(f"Classified as known {rf_label} attack")
        elif is_anomaly[i]:
            verdicts.append("unknown_suspicious")
            reasons.append("Predicted normal by classifier, but flagged anomalous")
        else:
            verdicts.append("normal")
            reasons.append("Normal — consistent across both models")
        confidences.append(rf_confidence[i])

    return pd.DataFrame({
        "verdict": verdicts,
        "reason": reasons,
        "classifier_confidence": confidences
    })


def preprocess_uploaded(df, scaler, feature_encoders):
    """Encode categorical cols and scale, matching training pipeline."""
    df = df.copy()
    for col in CATEGORICAL_COLS:
        le = feature_encoders[col]
        # Map unseen categories to the first known class rather than crashing
        known = set(le.classes_)
        df[col] = df[col].apply(lambda v: v if v in known else le.classes_[0])
        df[col] = le.transform(df[col])

    X = df[COLUMNS]
    X_scaled = scaler.transform(X)
    return pd.DataFrame(X_scaled, columns=COLUMNS)



st.title("🛡️ Cybersecurity Network Threat & Intrusion Profiler")
st.caption("Hybrid ML system — Random Forest classifier + Isolation Forest anomaly detector, trained on NSL-KDD")

try:
    rf, iso_forest, scaler, label_encoder, feature_encoders = load_models()
    class_names = label_encoder.classes_
except FileNotFoundError as e:
    st.error(f"Could not find a model file: {e}. Make sure all .pkl files are in the same folder as app.py.")
    st.stop()

st.sidebar.header("About")
st.sidebar.write(
    "This dashboard runs uploaded network traffic through two models:\n\n"
    "1. **Random Forest** — recognizes known attack types (DoS, Probe, R2L, U2R)\n"
    "2. **Isolation Forest** — flags traffic that looks abnormal, even if the "
    "classifier calls it 'normal'\n\n"
    "A record is flagged **unknown_suspicious** when the classifier says normal "
    "but the anomaly detector disagrees — this catches attack types the "
    "classifier wasn't trained to recognize."
)

st.subheader("1. Upload traffic data")
st.write(
    "Upload a CSV in NSL-KDD format (41 feature columns, no header row, "
    "same order as the original dataset). You can use a slice of `KDDTest+.txt` "
    "to try it out."
)

uploaded_file = st.file_uploader("Upload CSV", type=["csv", "txt"])

if uploaded_file is not None:
    try:
        raw = pd.read_csv(uploaded_file, header=None)
        # Accept files with or without the trailing label columns
        if raw.shape[1] >= 41:
            raw = raw.iloc[:, :41]
            raw.columns = COLUMNS
        else:
            st.error(f"Expected at least 41 columns, got {raw.shape[1]}.")
            st.stop()
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    st.write(f"Loaded **{raw.shape[0]}** connection records.")
    with st.expander("Preview raw data"):
        st.dataframe(raw.head(20))

    if st.button("Run threat analysis", type="primary"):
        with st.spinner("Running hybrid detection..."):
            X_scaled = preprocess_uploaded(raw, scaler, feature_encoders)
            results = hybrid_predict(X_scaled, rf, iso_forest, label_encoder, class_names)

        st.subheader("2. Results")

        col1, col2, col3, col4 = st.columns(4)
        n_total = len(results)
        n_normal = (results['verdict'] == 'normal').sum()
        n_known_attack = ((results['verdict'] != 'normal') & (results['verdict'] != 'unknown_suspicious')).sum()
        n_suspicious = (results['verdict'] == 'unknown_suspicious').sum()

        col1.metric("Total connections", n_total)
        col2.metric("Normal", n_normal)
        col3.metric("Known attacks", n_known_attack)
        col4.metric("Unknown / suspicious", n_suspicious)

        # ── Verdict breakdown chart ──
        st.subheader("Verdict breakdown")
        verdict_counts = results['verdict'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = {'normal': '#4CAF50', 'dos': '#F44336', 'probe': '#FF9800',
                  'r2l': '#9C27B0', 'u2r': '#B71C1C', 'unknown_suspicious': '#FFC107'}
        bar_colors = [colors.get(v, '#607D8B') for v in verdict_counts.index]
        ax.bar(verdict_counts.index, verdict_counts.values, color=bar_colors)
        ax.set_ylabel("Number of connections")
        ax.set_xlabel("Verdict")
        plt.xticks(rotation=20)
        st.pyplot(fig)

        # ── Flagged connections table ──
        st.subheader("Flagged connections (attacks + suspicious)")
        flagged = results[results['verdict'] != 'normal'].sort_values(
            by='classifier_confidence', ascending=False
        )
        if len(flagged) > 0:
            st.dataframe(flagged, use_container_width=True)
            csv = flagged.to_csv(index=False).encode('utf-8')
            st.download_button("Download flagged results as CSV", csv, "flagged_connections.csv", "text/csv")
        else:
            st.success("No attacks or suspicious traffic detected in this file.")

        # ── Full results ──
        with st.expander("View all results"):
            st.dataframe(results, use_container_width=True)
else:
    st.info("Upload a CSV file above to begin analysis.")
