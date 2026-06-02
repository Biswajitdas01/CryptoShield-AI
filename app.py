"""
app.py  –  CryptoShield AI  –  Main Streamlit Application
"""
import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from src.preprocessing import (
    preprocess_pipeline, get_missing_value_report,
    get_summary_stats, NUMERIC_FEATURES, BINARY_FEATURES, TARGET,
)
from src.train import train_random_forest, train_xgboost, compare_models, HAS_XGB
from src.predict import predict_dataframe, risk_summary, get_top_risky, explain_transaction
from src.explainability import (
    plot_feature_importance, plot_shap_summary, plot_shap_waterfall_single,
    plot_correlation_matrix, get_shap_values, HAS_SHAP,
)
from src.report_generator import generate_pdf_report

# ════════════════════════════════════════════
#  PAGE CONFIG
# ════════════════════════════════════════════
st.set_page_config(
    page_title="CryptoShield AI",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════
#  GLOBAL CSS
# ════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700&display=swap');

:root {
    --bg:      #0d1117;
    --surface: #161b22;
    --border:  #30363d;
    --accent:  #00ff88;
    --accent2: #1f6feb;
    --danger:  #f85149;
    --warn:    #d29922;
    --text:    #e6edf3;
    --muted:   #8b949e;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Exo 2', sans-serif;
}

/* sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* metric cards */
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 18px;
}
[data-testid="stMetricLabel"]  { color: var(--muted) !important; font-size: 12px; }
[data-testid="stMetricValue"]  { color: var(--accent) !important; font-family: 'Share Tech Mono'; font-size: 28px; }
[data-testid="stMetricDelta"]  { font-size: 11px; }

/* dataframes */
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 6px; }

/* file uploader */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--accent2) !important;
    border-radius: 8px;
    background: var(--surface) !important;
}

/* buttons */
.stButton > button {
    background: linear-gradient(135deg, #00ff88 0%, #1f6feb 100%) !important;
    color: #0d1117 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Exo 2', sans-serif !important;
    letter-spacing: 0.5px;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88; }

/* tabs */
.stTabs [data-baseweb="tab"] {
    background: var(--surface);
    color: var(--muted) !important;
    border-radius: 6px 6px 0 0;
    border: 1px solid var(--border);
    padding: 6px 16px;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: var(--border) !important;
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* expander */
.streamlit-expanderHeader {
    background: var(--surface) !important;
    color: var(--accent) !important;
    border-radius: 6px;
    border: 1px solid var(--border) !important;
}

/* progress */
.stProgress > div > div { background: var(--accent) !important; }

/* selectbox / slider */
.stSelectbox > div > div, .stSlider { color: var(--text) !important; }

/* scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════
#  LOGO / HEADER
# ════════════════════════════════════════════
def render_logo():
    st.markdown("""
    <div style="display:flex;align-items:center;gap:14px;padding:8px 0 18px 0;border-bottom:1px solid #30363d;margin-bottom:18px;">
      <div style="font-size:42px;line-height:1;">🛡</div>
      <div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:24px;color:#00ff88;letter-spacing:2px;">CRYPTOSHIELD AI</div>
        <div style="font-size:11px;color:#8b949e;letter-spacing:3px;text-transform:uppercase;">Cryptocurrency Scam Detection System</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

def section_header(icon, title, subtitle=""):
    st.markdown(f"""
    <div style="margin:8px 0 16px 0;">
      <div style="font-size:20px;font-weight:700;color:#e6edf3;">{icon}&nbsp;{title}</div>
      {"<div style='font-size:12px;color:#8b949e;margin-top:2px;'>"+subtitle+"</div>" if subtitle else ""}
      <hr style="border:none;border-top:1px solid #30363d;margin-top:8px;">
    </div>
    """, unsafe_allow_html=True)

def card(content_html):
    st.markdown(f"""
    <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;
                padding:16px;margin-bottom:12px;">{content_html}</div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════
#  SIDEBAR NAV
# ════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='font-family:"Share Tech Mono",monospace;font-size:18px;color:#00ff88;
                letter-spacing:2px;padding:8px 0 16px 0;border-bottom:1px solid #30363d;'>
    🛡 CRYPTOSHIELD
    </div>""", unsafe_allow_html=True)

    page = st.radio("", [
        "🏠  Home & Upload",
        "🔬  Data Analysis",
        "🤖  Model Training",
        "⚠️  Scam Detection",
        "📊  Dashboard",
        "🧠  Explainable AI",
        "📄  PDF Report",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("<div style='font-size:11px;color:#8b949e;'>SESSION STATE</div>", unsafe_allow_html=True)
    has_data   = "df" in st.session_state
    has_model  = "rf_metrics" in st.session_state
    has_preds  = "result_df" in st.session_state
    st.markdown(f"{'✅' if has_data else '⭕'} Dataset loaded", unsafe_allow_html=True)
    st.markdown(f"{'✅' if has_model else '⭕'} Model trained", unsafe_allow_html=True)
    st.markdown(f"{'✅' if has_preds else '⭕'} Predictions ready", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div style='font-size:10px;color:#8b949e;'>v1.0.0 · MIT License<br>CryptoShield AI © 2024</div>",
                unsafe_allow_html=True)

# ════════════════════════════════════════════
#  DARK PLOTLY THEME HELPER
# ════════════════════════════════════════════
def dark_layout(fig, title="", height=400):
    fig.update_layout(
        title=dict(text=title, font=dict(color="#e6edf3", size=14)),
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font=dict(color="#e6edf3"),
        xaxis=dict(gridcolor="#30363d", zerolinecolor="#30363d"),
        yaxis=dict(gridcolor="#30363d", zerolinecolor="#30363d"),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
        margin=dict(l=20, r=20, t=40, b=20),
        height=height,
    )
    return fig

# ════════════════════════════════════════════
#  PAGE: HOME & UPLOAD
# ════════════════════════════════════════════
if "🏠" in page:
    render_logo()
    section_header("📁", "Dataset Upload", "Upload a CSV file to begin fraud detection analysis")

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader(
            "Drop your cryptocurrency transaction CSV here",
            type=["csv"], help="Required columns: amount_usd, gas_fee_usd, is_fraud, etc.")

        if uploaded:
            with st.spinner("Reading and validating dataset..."):
                df = pd.read_csv(uploaded)
                st.session_state["df"] = df
                st.success(f"✅ Loaded **{len(df):,}** rows × **{df.shape[1]}** columns")

        # sample data loader
        st.markdown("**Or use the built-in sample dataset:**")
        if st.button("⚡ Load Sample Dataset (4,800 transactions)"):
            path = "data/crypto_transactions.csv"
            if os.path.exists(path):
                df = pd.read_csv(path)
                st.session_state["df"] = df
                st.success(f"✅ Sample dataset loaded: {len(df):,} rows")
            else:
                st.warning("Run `python data/generate_sample.py` first.")

    with col2:
        card("""
        <div style='font-size:13px;color:#8b949e;'>
        <div style='color:#00ff88;font-weight:700;margin-bottom:8px;'>📋 Required Columns</div>
        • amount_usd<br>• gas_fee_usd<br>• transaction_speed_s<br>
        • num_transactions_24h<br>• wallet_age_days<br>• unique_receivers_30d<br>
        • mixing_service_flag<br>• rapid_movement_flag<br>• blacklist_interaction<br>
        • is_fraud (0 / 1)<br><br>
        <div style='color:#d29922;'>Optional: sender_wallet, receiver_wallet, timestamp, token_type, network</div>
        </div>""")

    if "df" in st.session_state:
        df = st.session_state["df"]
        st.markdown("---")
        section_header("👁", "Dataset Preview")
        st.dataframe(df.head(15), use_container_width=True)

        section_header("📊", "Summary Statistics")
        stats = get_summary_stats(df)
        st.dataframe(stats, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Rows",    f"{len(df):,}")
        c2.metric("Columns",       df.shape[1])
        c3.metric("Fraud Rows",    f"{df[TARGET].sum():,}" if TARGET in df.columns else "N/A")
        c4.metric("Fraud Rate",    f"{df[TARGET].mean()*100:.1f}%" if TARGET in df.columns else "N/A")

# ════════════════════════════════════════════
#  PAGE: DATA ANALYSIS
# ════════════════════════════════════════════
elif "🔬" in page:
    render_logo()
    section_header("🔬", "Data Analysis", "Explore distributions, correlations, and missing values")

    if "df" not in st.session_state:
        st.warning("⚠️ Please upload a dataset first.")
        st.stop()

    df = st.session_state["df"]
    tabs = st.tabs(["Missing Values", "Distributions", "Correlation Matrix", "Behaviour Analysis"])

    # ── Tab 1: Missing Values
    with tabs[0]:
        mv = get_missing_value_report(df)
        col1, col2 = st.columns([3, 2])
        with col1:
            st.dataframe(mv, use_container_width=True)
        with col2:
            nonzero = mv[mv["Missing Count"] > 0]
            if len(nonzero):
                fig = px.bar(nonzero, x="Column", y="Missing %",
                             color="Missing %", color_continuous_scale="reds")
                dark_layout(fig, "Missing Value Distribution", 320)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("🎉 No missing values detected!")

    # ── Tab 2: Distributions
    with tabs[1]:
        num_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
        selected = st.selectbox("Select feature", num_cols)
        if selected:
            col1, col2 = st.columns(2)
            with col1:
                if TARGET in df.columns:
                    fig = px.histogram(df, x=selected, color=TARGET,
                                       color_discrete_map={0:"#1f6feb", 1:"#f85149"},
                                       barmode="overlay", nbins=50, opacity=0.75)
                    dark_layout(fig, f"Distribution of {selected} by Fraud Label", 360)
                else:
                    fig = px.histogram(df, x=selected, nbins=50, color_discrete_sequence=["#00ff88"])
                    dark_layout(fig, f"Distribution of {selected}", 360)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                if TARGET in df.columns:
                    fig2 = px.box(df, x=TARGET, y=selected,
                                  color=TARGET, color_discrete_map={0:"#1f6feb", 1:"#f85149"})
                    dark_layout(fig2, f"Box Plot: {selected}", 360)
                    st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 3: Correlation
    with tabs[2]:
        fig = plot_correlation_matrix(df)
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 4: Behaviour
    with tabs[3]:
        if TARGET in df.columns:
            fraud_df  = df[df[TARGET] == 1]
            legit_df  = df[df[TARGET] == 0]
            col1, col2 = st.columns(2)
            with col1:
                means = pd.DataFrame({
                    "Legitimate": legit_df[NUMERIC_FEATURES].mean(),
                    "Fraud":      fraud_df[NUMERIC_FEATURES].mean(),
                }).reset_index().rename(columns={"index":"Feature"})
                fig = go.Figure()
                fig.add_trace(go.Bar(name="Legitimate", x=means["Feature"], y=means["Legitimate"],
                                     marker_color="#1f6feb"))
                fig.add_trace(go.Bar(name="Fraud", x=means["Feature"], y=means["Fraud"],
                                     marker_color="#f85149"))
                dark_layout(fig, "Mean Feature Values: Legit vs Fraud", 420)
                fig.update_layout(barmode="group", xaxis_tickangle=-40)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                binary_cols = [c for c in BINARY_FEATURES if c in df.columns]
                fraud_rates = {c: fraud_df[c].mean() for c in binary_cols}
                legit_rates = {c: legit_df[c].mean() for c in binary_cols}
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(name="Legitimate", x=list(legit_rates.keys()),
                                      y=list(legit_rates.values()), marker_color="#1f6feb"))
                fig2.add_trace(go.Bar(name="Fraud", x=list(fraud_rates.keys()),
                                      y=list(fraud_rates.values()), marker_color="#f85149"))
                dark_layout(fig2, "Binary Flag Rates: Legit vs Fraud", 420)
                fig2.update_layout(barmode="group", xaxis_tickangle=-40, yaxis_tickformat=".0%")
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No 'is_fraud' column found for behaviour comparison.")

# ════════════════════════════════════════════
#  PAGE: MODEL TRAINING
# ════════════════════════════════════════════
elif "🤖" in page:
    render_logo()
    section_header("🤖", "Model Training", "Train and evaluate ML models on your dataset")

    if "df" not in st.session_state:
        st.warning("⚠️ Please upload a dataset first.")
        st.stop()

    df = st.session_state["df"]

    with st.expander("⚙️ Training Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            n_est = st.slider("N Estimators", 50, 500, 200, step=50)
            test_sz = st.slider("Test Split %", 10, 40, 20) / 100
        with col2:
            max_d = st.selectbox("Max Depth (RF)", [None, 5, 10, 15, 20])
            lr = st.slider("XGBoost LR", 0.01, 0.3, 0.1, step=0.01)
        with col3:
            models_to_train = st.multiselect(
                "Models to Train",
                ["Random Forest", "XGBoost"] if HAS_XGB else ["Random Forest"],
                default=["Random Forest", "XGBoost"] if HAS_XGB else ["Random Forest"],
            )

    if st.button("🚀 Train Selected Models"):
        with st.spinner("Preprocessing data..."):
            pipeline = preprocess_pipeline(df, test_size=test_sz)
            st.session_state["pipeline"] = pipeline

        vr = pipeline["validation_report"]
        if vr["errors"]:
            for e in vr["errors"]:
                st.error(e)
            st.stop()
        if vr["warnings"]:
            for w in vr["warnings"]:
                st.warning(w)

        model_metrics = {}

        if "Random Forest" in models_to_train:
            with st.spinner("Training Random Forest..."):
                rf_m = train_random_forest(pipeline, n_estimators=n_est, max_depth=max_d)
                st.session_state["rf_metrics"] = rf_m
                model_metrics["Random Forest"] = rf_m
                st.success(f"✅ Random Forest trained in {rf_m['training_time_s']}s  |  ROC-AUC: {rf_m['roc_auc']}")

        if "XGBoost" in models_to_train and HAS_XGB:
            with st.spinner("Training XGBoost..."):
                xgb_m = train_xgboost(pipeline, n_estimators=n_est, learning_rate=lr)
                st.session_state["xgb_metrics"] = xgb_m
                model_metrics["XGBoost"] = xgb_m
                st.success(f"✅ XGBoost trained in {xgb_m['training_time_s']}s  |  ROC-AUC: {xgb_m['roc_auc']}")

        st.session_state["model_metrics"] = model_metrics

    # ── Display results if trained
    if "model_metrics" in st.session_state:
        model_metrics = st.session_state["model_metrics"]
        rf_m  = st.session_state.get("rf_metrics", {})
        xgb_m = st.session_state.get("xgb_metrics", {})

        section_header("📈", "Model Performance")

        # comparison table
        cmp_df = compare_models(rf_m, xgb_m)
        if not cmp_df.empty:
            st.dataframe(cmp_df.set_index("Model"), use_container_width=True)

        # ROC curves
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_shape(type="line", x0=0, x1=1, y0=0, y1=1,
                          line=dict(color="#30363d", dash="dash"))
            colors_ = {"Random Forest": "#00ff88", "XGBoost": "#1f6feb"}
            for name, m in model_metrics.items():
                if "fpr" in m:
                    fig.add_trace(go.Scatter(
                        x=m["fpr"], y=m["tpr"], mode="lines",
                        name=f"{name} (AUC={m['roc_auc']})",
                        line=dict(color=colors_.get(name, "#00ff88"), width=2),
                    ))
            dark_layout(fig, "ROC Curves", 380)
            fig.update_layout(
                xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # confusion matrix heatmap for best model
            best = rf_m if "confusion_matrix" in rf_m else xgb_m
            if "confusion_matrix" in best:
                cm = np.array(best["confusion_matrix"])
                fig2 = go.Figure(go.Heatmap(
                    z=cm, x=["Pred Legit","Pred Fraud"], y=["Actual Legit","Actual Fraud"],
                    colorscale=[[0,"#161b22"],[1,"#00ff88"]],
                    text=cm, texttemplate="%{text}", textfont=dict(size=18),
                    showscale=False,
                ))
                dark_layout(fig2, f"Confusion Matrix – {best.get('model_name','Model')}", 380)
                st.plotly_chart(fig2, use_container_width=True)

        # metric cards
        best = rf_m if rf_m else xgb_m
        if best and "accuracy" in best:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Accuracy",  f"{best['accuracy']:.2%}")
            c2.metric("Precision", f"{best['precision']:.2%}")
            c3.metric("Recall",    f"{best['recall']:.2%}")
            c4.metric("F1 Score",  f"{best['f1']:.2%}")
            c5.metric("ROC-AUC",   f"{best['roc_auc']:.4f}")

# ════════════════════════════════════════════
#  PAGE: SCAM DETECTION
# ════════════════════════════════════════════
elif "⚠️" in page:
    render_logo()
    section_header("⚠️", "Scam Detection", "Score every transaction and flag high-risk activity")

    if "pipeline" not in st.session_state or "rf_metrics" not in st.session_state:
        st.warning("⚠️ Please train a model first.")
        st.stop()

    pipeline = st.session_state["pipeline"]
    rf_m     = st.session_state["rf_metrics"]
    model    = rf_m["model"]

    if st.button("🔍 Run Scam Detection on Full Dataset"):
        with st.spinner("Scoring all transactions..."):
            result_df = predict_dataframe(
                model,
                pipeline["X_test"],
                pipeline["feature_names"],
            )
            # re-attach y_test for comparison
            result_df["actual_fraud"] = pipeline["y_test"].values
            st.session_state["result_df"] = result_df
            summary = risk_summary(result_df)
            st.session_state["detection_summary"] = summary
            st.success(f"✅ Scored {len(result_df):,} transactions")

    if "result_df" in st.session_state:
        result_df = st.session_state["result_df"]
        summary   = st.session_state["detection_summary"]

        # kpi row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Transactions Scored", f"{summary['total_transactions']:,}")
        c2.metric("Fraud Detected",      f"{summary['fraud_detected']:,}", f"{summary['fraud_pct']}%")
        c3.metric("High Risk",           f"{summary['high_risk']:,}")
        c4.metric("Avg Risk Score",      f"{summary['avg_risk_score']}")

        col1, col2 = st.columns(2)
        with col1:
            # risk distribution
            counts = result_df["risk_label"].value_counts()
            color_map = {"🔴 High Risk":"#f85149","🟡 Medium Risk":"#d29922","🟢 Low Risk":"#00ff88"}
            fig = go.Figure(go.Pie(
                labels=counts.index, values=counts.values,
                marker=dict(colors=[color_map.get(l,"#8b949e") for l in counts.index]),
                hole=0.45, textfont=dict(size=13),
            ))
            dark_layout(fig, "Risk Distribution", 340)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # risk score histogram
            fig2 = px.histogram(result_df, x="risk_score", nbins=40,
                                color_discrete_sequence=["#1f6feb"])
            fig2.add_vline(x=40, line_dash="dash", line_color="#d29922", annotation_text="Medium")
            fig2.add_vline(x=70, line_dash="dash", line_color="#f85149", annotation_text="High")
            dark_layout(fig2, "Risk Score Distribution", 340)
            st.plotly_chart(fig2, use_container_width=True)

        section_header("🚨", "Top High-Risk Transactions")
        top = get_top_risky(result_df, n=30)
        display_cols = [c for c in ["risk_score","risk_label","fraud_probability",
                                     "predicted_fraud","actual_fraud"] if c in top.columns]
        st.dataframe(top[display_cols], use_container_width=True)

        # individual transaction inspector
        section_header("🔎", "Transaction Inspector")
        idx = st.number_input("Select transaction index", 0, len(result_df)-1, 0)
        row = result_df.iloc[idx]
        exp = explain_transaction(row, rf_m.get("feature_importance", {}))

        color_map2 = {"🔴 High Risk":"#f85149","🟡 Medium Risk":"#d29922","🟢 Low Risk":"#00ff88"}
        label_color = color_map2.get(exp["risk_label"], "#00ff88")
        card(f"""
        <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;'>
          <div>
            <div style='color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:1px;'>Risk Label</div>
            <div style='font-size:22px;font-weight:700;color:{label_color};'>{exp['risk_label']}</div>
          </div>
          <div>
            <div style='color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:1px;'>Risk Score</div>
            <div style='font-family:"Share Tech Mono",monospace;font-size:22px;color:{label_color};'>{exp['risk_score']} / 100</div>
          </div>
          <div>
            <div style='color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:1px;'>Fraud Probability</div>
            <div style='font-family:"Share Tech Mono",monospace;font-size:22px;color:{label_color};'>{exp['fraud_probability']:.2%}</div>
          </div>
        </div>
        """)

        st.markdown("**Top Risk Factors:**")
        for f in exp["top_risk_factors"]:
            pct = int(f["importance"] * 100)
            st.markdown(f"""
            <div style='margin:4px 0;'>
              <span style='color:#8b949e;font-size:12px;'>{f['feature']}</span>
              <span style='color:#e6edf3;float:right;font-size:12px;'>val: {f['value']}</span>
              <div style='background:#161b22;border-radius:4px;margin-top:2px;'>
                <div style='background:#1f6feb;height:6px;border-radius:4px;width:{min(pct*5,100)}%;'></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════
#  PAGE: DASHBOARD
# ════════════════════════════════════════════
elif "📊" in page:
    render_logo()
    section_header("📊", "Analytics Dashboard", "Interactive overview of fraud patterns and trends")

    if "result_df" not in st.session_state or "df" not in st.session_state:
        st.warning("⚠️ Please run Scam Detection first.")
        st.stop()

    df        = st.session_state["df"]
    result_df = st.session_state["result_df"]
    summary   = st.session_state.get("detection_summary", {})
    rf_m      = st.session_state.get("rf_metrics", {})

    # KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Transactions", f"{summary.get('total_transactions',0):,}")
    c2.metric("Fraud %",            f"{summary.get('fraud_pct',0)}%")
    c3.metric("High Risk Wallets",  f"{summary.get('high_risk',0):,}")
    c4.metric("Avg Risk Score",     f"{summary.get('avg_risk_score',0)}")
    c5.metric("Best ROC-AUC",       f"{rf_m.get('roc_auc','N/A')}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    # fraud by token type
    with col1:
        if "token_type" in df.columns and TARGET in df.columns:
            tf = df.groupby("token_type")[TARGET].agg(["sum","count"]).reset_index()
            tf.columns = ["Token","Fraud","Total"]
            tf["Rate"] = (tf["Fraud"]/tf["Total"]*100).round(1)
            fig = px.bar(tf, x="Token", y="Rate", color="Rate",
                         color_continuous_scale="reds", text="Rate")
            dark_layout(fig, "Fraud Rate by Token Type (%)", 340)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "network" in df.columns and TARGET in df.columns:
            nf = df.groupby("network")[TARGET].agg(["sum","count"]).reset_index()
            nf.columns = ["Network","Fraud","Total"]
            nf["Rate"] = (nf["Fraud"]/nf["Total"]*100).round(1)
            fig2 = go.Figure(go.Pie(
                labels=nf["Network"], values=nf["Fraud"], hole=0.4,
                marker=dict(colors=["#f85149","#d29922","#1f6feb","#00ff88","#8b949e"]),
            ))
            dark_layout(fig2, "Fraud Count by Network", 340)
            st.plotly_chart(fig2, use_container_width=True)

    # trend analysis
    col3, col4 = st.columns(2)
    with col3:
        if "timestamp" in df.columns and TARGET in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["month"] = df["timestamp"].dt.to_period("M").astype(str)
            ts = df.groupby("month")[TARGET].agg(["sum","count"]).reset_index()
            ts.columns = ["Month","Fraud","Total"]
            ts["Rate"] = (ts["Fraud"]/ts["Total"]*100).round(2)
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=ts["Month"], y=ts["Fraud"],
                                      mode="lines+markers", name="Fraud Count",
                                      line=dict(color="#f85149", width=2),
                                      marker=dict(size=6)))
            dark_layout(fig3, "Fraud Trend Over Time", 340)
            fig3.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # amount vs risk score scatter
        if "amount_usd" in result_df.columns:
            samp = result_df.sample(min(500, len(result_df)), random_state=42)
            fig4 = px.scatter(samp, x="amount_usd", y="risk_score",
                              color="risk_label",
                              color_discrete_map={
                                  "🔴 High Risk":"#f85149",
                                  "🟡 Medium Risk":"#d29922",
                                  "🟢 Low Risk":"#00ff88"},
                              opacity=0.7, log_x=True)
            dark_layout(fig4, "Transaction Amount vs Risk Score", 340)
            st.plotly_chart(fig4, use_container_width=True)

    # binary flags heatmap
    section_header("🔥", "Risk Flag Heatmap")
    binary_cols = [c for c in BINARY_FEATURES if c in df.columns]
    if TARGET in df.columns and binary_cols:
        heat_data = df.groupby(TARGET)[binary_cols].mean()
        heat_data.index = ["Legitimate","Fraud"]
        fig5 = go.Figure(go.Heatmap(
            z=heat_data.values, x=binary_cols, y=heat_data.index,
            colorscale="YlOrRd", text=heat_data.values.round(2),
            texttemplate="%{text}", textfont=dict(size=11),
        ))
        dark_layout(fig5, "Average Binary Flag Rate: Legitimate vs Fraud", 250)
        st.plotly_chart(fig5, use_container_width=True)

# ════════════════════════════════════════════
#  PAGE: EXPLAINABLE AI
# ════════════════════════════════════════════
elif "🧠" in page:
    render_logo()
    section_header("🧠", "Explainable AI", "Understand what drives fraud predictions")

    if "rf_metrics" not in st.session_state:
        st.warning("⚠️ Please train a model first.")
        st.stop()

    rf_m  = st.session_state["rf_metrics"]
    xgb_m = st.session_state.get("xgb_metrics", {})

    tabs = st.tabs(["Feature Importance", "SHAP Analysis", "Single Prediction Explanation"])

    with tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            fig = plot_feature_importance(rf_m["feature_importance"], title="Random Forest – Feature Importance")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            if xgb_m and "feature_importance" in xgb_m:
                fig2 = plot_feature_importance(xgb_m["feature_importance"], title="XGBoost – Feature Importance")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Train XGBoost to compare feature importance.")

    with tabs[1]:
        if not HAS_SHAP:
            st.warning("Install SHAP: `pip install shap`")
        else:
            pipeline = st.session_state.get("pipeline")
            if pipeline:
                with st.spinner("Computing SHAP values (may take ~10s)..."):
                    sv, sample = get_shap_values(rf_m["model"], pipeline["X_train"])
                if sv is not None:
                    fig3 = plot_shap_summary(sv, sample)
                    if fig3: st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.warning("SHAP computation failed for this model type.")

    with tabs[2]:
        if "pipeline" in st.session_state:
            pipeline = st.session_state["pipeline"]
            idx2 = st.slider("Transaction index (from test set)", 0, len(pipeline["X_test"])-1, 0)
            if HAS_SHAP:
                with st.spinner("Computing SHAP for transaction..."):
                    sv2, samp2 = get_shap_values(rf_m["model"],
                                                  pipeline["X_test"].iloc[[idx2]])
                if sv2 is not None:
                    fig4 = plot_shap_waterfall_single(sv2, samp2, 0)
                    if fig4: st.plotly_chart(fig4, use_container_width=True)
            # also show simple feature explanation
            row = pipeline["X_test"].iloc[idx2].copy()
            if "result_df" in st.session_state:
                result_df = st.session_state["result_df"]
                if idx2 < len(result_df):
                    row2 = result_df.iloc[idx2]
                    exp = explain_transaction(row2, rf_m["feature_importance"])
                    st.json(exp)

# ════════════════════════════════════════════
#  PAGE: PDF REPORT
# ════════════════════════════════════════════
elif "📄" in page:
    render_logo()
    section_header("📄", "PDF Report Generator", "Export a professional intelligence report")

    if "detection_summary" not in st.session_state:
        st.warning("⚠️ Please run Scam Detection first to generate a report.")
        st.stop()

    df      = st.session_state.get("df", pd.DataFrame())
    summary = st.session_state["detection_summary"]
    rf_m    = st.session_state.get("rf_metrics", {})
    xgb_m   = st.session_state.get("xgb_metrics", {})
    model_metrics = {"Random Forest": rf_m}
    if xgb_m: model_metrics["XGBoost"] = xgb_m

    dataset_info = {
        "Total Rows":          len(df),
        "Total Columns":       df.shape[1],
        "Fraud Rate (%)":      f"{df[TARGET].mean()*100:.2f}" if TARGET in df.columns else "N/A",
        "Date Range":          f"{df['timestamp'].min()} → {df['timestamp'].max()}" if "timestamp" in df.columns else "N/A",
        "Report Generated At": datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
    }

    card("""
    <div style='color:#e6edf3;font-size:13px;'>
    📋 The PDF report will include:<br><br>
    • Executive Summary<br>
    • Dataset Overview & Statistics<br>
    • Fraud Detection Results<br>
    • Risk Classification Table<br>
    • Model Performance Metrics (Accuracy, Precision, Recall, F1, ROC-AUC)<br>
    • Methodology & Disclaimer
    </div>
    """)

    if st.button("📥 Generate PDF Report"):
        with st.spinner("Generating professional PDF report..."):
            out = generate_pdf_report(summary, model_metrics, dataset_info)

        with open(out, "rb") as f:
            pdf_bytes = f.read()

        st.success(f"✅ Report generated: `{out}`")
        st.download_button(
            label="⬇️ Download PDF Report",
            data=pdf_bytes,
            file_name=f"CryptoShield_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
        )
