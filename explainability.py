"""
src/explainability.py
CryptoShield AI – Explainability: SHAP values and feature analysis.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Optional

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

DARK_BG = "#0d1117"
ACCENT  = "#00ff88"

def get_shap_values(model, X: pd.DataFrame, max_samples: int = 300):
    if not HAS_SHAP:
        return None, None
    sample = X.sample(min(max_samples, len(X)), random_state=42)
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(sample)
        if isinstance(sv, list):          # RF returns list [class0, class1]
            sv = sv[1]
        return sv, sample
    except Exception:
        return None, None

def plot_feature_importance(feature_importance: Dict[str, float],
                             top_n: int = 15, title: str = "Feature Importance") -> go.Figure:
    fi = pd.Series(feature_importance).nlargest(top_n).sort_values()
    colors = [ACCENT if i >= len(fi) - 3 else "#1f6feb" for i in range(len(fi))]
    fig = go.Figure(go.Bar(
        x=fi.values, y=fi.index, orientation="h",
        marker=dict(color=colors, line=dict(color=ACCENT, width=0.5)),
        text=[f"{v:.3f}" for v in fi.values], textposition="outside",
        textfont=dict(color="#e6edf3", size=11),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#e6edf3", size=16)),
        paper_bgcolor=DARK_BG, plot_bgcolor="#161b22",
        xaxis=dict(showgrid=True, gridcolor="#30363d", color="#e6edf3", title="Importance Score"),
        yaxis=dict(showgrid=False, color="#e6edf3"),
        margin=dict(l=20, r=40, t=50, b=20), height=420,
    )
    return fig

def plot_shap_summary(shap_values, sample: pd.DataFrame) -> Optional[go.Figure]:
    if shap_values is None:
        return None
    mean_abs = np.abs(shap_values).mean(axis=0)
    fi = pd.Series(mean_abs, index=sample.columns).nlargest(15).sort_values()
    fig = go.Figure(go.Bar(
        x=fi.values, y=fi.index, orientation="h",
        marker=dict(color=fi.values, colorscale="Plasma",
                    line=dict(color=ACCENT, width=0.5)),
        text=[f"{v:.4f}" for v in fi.values], textposition="outside",
        textfont=dict(color="#e6edf3", size=10),
    ))
    fig.update_layout(
        title=dict(text="SHAP – Mean |SHAP Value| per Feature", font=dict(color="#e6edf3", size=15)),
        paper_bgcolor=DARK_BG, plot_bgcolor="#161b22",
        xaxis=dict(showgrid=True, gridcolor="#30363d", color="#e6edf3", title="Mean |SHAP|"),
        yaxis=dict(showgrid=False, color="#e6edf3"),
        margin=dict(l=20, r=40, t=50, b=20), height=420,
    )
    return fig

def plot_shap_waterfall_single(shap_values, sample: pd.DataFrame, idx: int = 0) -> Optional[go.Figure]:
    if shap_values is None or idx >= len(shap_values):
        return None
    sv = shap_values[idx]
    feat = sample.columns.tolist()
    order = np.argsort(np.abs(sv))[::-1][:12]
    sv_top = sv[order]
    feat_top = [feat[i] for i in order]
    colors = [ACCENT if v > 0 else "#f85149" for v in sv_top]
    fig = go.Figure(go.Bar(
        x=sv_top, y=feat_top, orientation="h",
        marker=dict(color=colors),
        text=[f"{v:+.4f}" for v in sv_top], textposition="outside",
        textfont=dict(color="#e6edf3", size=10),
    ))
    fig.update_layout(
        title=dict(text=f"SHAP Explanation – Transaction #{idx}", font=dict(color="#e6edf3", size=14)),
        paper_bgcolor=DARK_BG, plot_bgcolor="#161b22",
        xaxis=dict(showgrid=True, gridcolor="#30363d", color="#e6edf3", zeroline=True, zerolinecolor="#30363d"),
        yaxis=dict(showgrid=False, color="#e6edf3"),
        margin=dict(l=20, r=40, t=50, b=20), height=380,
    )
    return fig

def plot_correlation_matrix(df: pd.DataFrame) -> go.Figure:
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr().round(2)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=corr.values.round(2), texttemplate="%{text}",
        textfont=dict(size=8), showscale=True,
        colorbar=dict(tickfont=dict(color="#e6edf3")),
    ))
    fig.update_layout(
        title=dict(text="Feature Correlation Matrix", font=dict(color="#e6edf3", size=15)),
        paper_bgcolor=DARK_BG, plot_bgcolor="#161b22",
        xaxis=dict(color="#e6edf3", tickangle=-45, tickfont=dict(size=9)),
        yaxis=dict(color="#e6edf3", tickfont=dict(size=9)),
        margin=dict(l=20, r=20, t=50, b=80), height=520,
    )
    return fig
