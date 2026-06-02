"""
src/report_generator.py
CryptoShield AI – Professional PDF report generation via ReportLab.
"""
from __future__ import annotations
import os, io
from datetime import datetime
from typing import Dict, Any, Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak,
)
from reportlab.platypus import Image as RLImage
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

os.makedirs("reports", exist_ok=True)

# ── palette
C_BG      = colors.HexColor("#0d1117")
C_ACCENT  = colors.HexColor("#00ff88")
C_CARD    = colors.HexColor("#161b22")
C_TEXT    = colors.HexColor("#e6edf3")
C_MUTED   = colors.HexColor("#8b949e")
C_DANGER  = colors.HexColor("#f85149")
C_WARN    = colors.HexColor("#d29922")
C_WHITE   = colors.white

def _styles():
    ss = getSampleStyleSheet()
    base = dict(fontName="Helvetica", textColor=C_TEXT, backColor=C_BG)
    title_s   = ParagraphStyle("title",   fontSize=26, alignment=TA_CENTER,
                                textColor=C_ACCENT, spaceAfter=6, fontName="Helvetica-Bold")
    sub_s     = ParagraphStyle("sub",     fontSize=11, alignment=TA_CENTER,
                                textColor=C_MUTED,  spaceAfter=4, fontName="Helvetica")
    h1_s      = ParagraphStyle("h1",      fontSize=16, textColor=C_ACCENT,
                                spaceBefore=14, spaceAfter=6, fontName="Helvetica-Bold")
    h2_s      = ParagraphStyle("h2",      fontSize=13, textColor=C_TEXT,
                                spaceBefore=10, spaceAfter=4, fontName="Helvetica-Bold")
    body_s    = ParagraphStyle("body",    fontSize=10, textColor=C_TEXT,
                                spaceAfter=4,  leading=14, fontName="Helvetica")
    caption_s = ParagraphStyle("caption", fontSize=8,  textColor=C_MUTED,
                                alignment=TA_CENTER,   fontName="Helvetica-Oblique")
    return title_s, sub_s, h1_s, h2_s, body_s, caption_s

def _metric_table(rows, col_widths=None):
    col_widths = col_widths or [80*mm, 60*mm]
    t = Table(rows, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1, 0), C_CARD),
        ("TEXTCOLOR",    (0,0), (-1, 0), C_ACCENT),
        ("FONTNAME",     (0,0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 9),
        ("TEXTCOLOR",    (0,1), (-1,-1), C_TEXT),
        ("BACKGROUND",   (0,1), (-1,-1), C_BG),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[C_BG, C_CARD]),
        ("GRID",         (0,0), (-1,-1), 0.4, C_CARD),
        ("ALIGN",        (1,0), (-1,-1), "CENTER"),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING",  (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING",   (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0), (-1,-1), 4),
    ]))
    return t

def generate_pdf_report(
    summary: Dict[str, Any],
    model_metrics: Dict[str, Any],
    dataset_info: Dict[str, Any],
    output_path: str = "reports/cryptoshield_report.pdf",
) -> str:
    ts_s, sub_s, h1_s, h2_s, body_s, cap_s = _styles()
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=18*mm, bottomMargin=18*mm,
    )
    story = []

    # ── cover
    story.append(Spacer(1, 20*mm))
    story.append(Paragraph("🛡 CryptoShield AI", ts_s))
    story.append(Paragraph("Cryptocurrency Scam Detection – Intelligence Report", sub_s))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}", sub_s))
    story.append(HRFlowable(width="100%", thickness=1, color=C_ACCENT, spaceAfter=10))
    story.append(Spacer(1, 8*mm))

    # ── exec summary
    story.append(Paragraph("Executive Summary", h1_s))
    story.append(HRFlowable(width="100%", thickness=0.3, color=C_CARD, spaceAfter=4))
    story.append(Paragraph(
        "This report presents the findings of CryptoShield AI's automated fraud detection analysis "
        "on the submitted cryptocurrency transaction dataset. Machine Learning models were trained "
        "and evaluated to identify high-risk transactions, suspicious wallet behaviour, and potential "
        "scam patterns. Results are presented with full metric transparency for regulatory and "
        "research use.", body_s))
    story.append(Spacer(1, 4*mm))

    # ── dataset info
    story.append(Paragraph("Dataset Overview", h1_s))
    story.append(HRFlowable(width="100%", thickness=0.3, color=C_CARD, spaceAfter=4))
    di_rows = [["Metric", "Value"]]
    for k, v in dataset_info.items():
        di_rows.append([str(k), str(v)])
    story.append(_metric_table(di_rows))
    story.append(Spacer(1, 6*mm))

    # ── detection results
    story.append(Paragraph("Fraud Detection Results", h1_s))
    story.append(HRFlowable(width="100%", thickness=0.3, color=C_CARD, spaceAfter=4))
    res_rows = [["Metric", "Value"]]
    fields = [
        ("Total Transactions Analysed", summary.get("total_transactions","N/A")),
        ("Fraud Transactions Detected",  summary.get("fraud_detected","N/A")),
        ("Fraud Percentage",             f"{summary.get('fraud_pct','N/A')} %"),
        ("High-Risk Transactions",       summary.get("high_risk","N/A")),
        ("Medium-Risk Transactions",     summary.get("medium_risk","N/A")),
        ("Low-Risk Transactions",        summary.get("low_risk","N/A")),
        ("Average Risk Score",           summary.get("avg_risk_score","N/A")),
    ]
    for label, val in fields:
        res_rows.append([label, str(val)])
    story.append(_metric_table(res_rows))
    story.append(Spacer(1, 6*mm))

    # ── model performance
    story.append(Paragraph("Model Performance Metrics", h1_s))
    story.append(HRFlowable(width="100%", thickness=0.3, color=C_CARD, spaceAfter=4))
    perf_rows = [["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]]
    for name, m in model_metrics.items():
        if "error" not in m:
            perf_rows.append([
                name, str(m.get("accuracy","-")), str(m.get("precision","-")),
                str(m.get("recall","-")), str(m.get("f1","-")), str(m.get("roc_auc","-")),
            ])
    t = Table(perf_rows, colWidths=[38*mm,26*mm,26*mm,24*mm,24*mm,26*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,0), C_CARD),
        ("TEXTCOLOR",    (0,0),(-1,0), C_ACCENT),
        ("FONTNAME",     (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",     (0,0),(-1,-1), 9),
        ("TEXTCOLOR",    (0,1),(-1,-1), C_TEXT),
        ("BACKGROUND",   (0,1),(-1,-1), C_BG),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[C_BG, C_CARD]),
        ("GRID",         (0,0),(-1,-1), 0.4, C_CARD),
        ("ALIGN",        (1,0),(-1,-1), "CENTER"),
        ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",   (0,0),(-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 6*mm))

    # ── risk analysis
    story.append(Paragraph("Risk Analysis", h1_s))
    story.append(HRFlowable(width="100%", thickness=0.3, color=C_CARD, spaceAfter=4))
    total = summary.get("total_transactions", 1) or 1
    risk_rows = [["Risk Level", "Count", "Percentage", "Action Required"]]
    risk_rows.append(["🔴 High Risk",   str(summary.get("high_risk",0)),
                       f"{summary.get('high_risk',0)/total*100:.1f}%",  "Immediate Review"])
    risk_rows.append(["🟡 Medium Risk", str(summary.get("medium_risk",0)),
                       f"{summary.get('medium_risk',0)/total*100:.1f}%","Monitor Closely"])
    risk_rows.append(["🟢 Low Risk",    str(summary.get("low_risk",0)),
                       f"{summary.get('low_risk',0)/total*100:.1f}%",   "No Action"])
    story.append(_metric_table(risk_rows, col_widths=[50*mm,35*mm,35*mm,50*mm]))
    story.append(Spacer(1, 6*mm))

    # ── disclaimer
    story.append(PageBreak())
    story.append(Paragraph("Methodology & Disclaimer", h1_s))
    story.append(HRFlowable(width="100%", thickness=0.3, color=C_CARD, spaceAfter=4))
    story.append(Paragraph(
        "CryptoShield AI uses a supervised Machine Learning pipeline (Random Forest + XGBoost) "
        "trained on labelled cryptocurrency transaction data. Features include transaction amounts, "
        "gas fees, wallet age, transaction velocity, and behavioural flags. "
        "Risk scores are derived from model posterior probabilities scaled to [0, 100]. "
        "This report is for informational purposes only and does not constitute legal or financial advice. "
        "Always consult a qualified professional before taking action on flagged transactions.", body_s))
    story.append(Spacer(1, 6*mm))
    story.append(HRFlowable(width="100%", thickness=1, color=C_ACCENT))
    story.append(Paragraph("© CryptoShield AI – Confidential Intelligence Report", cap_s))

    doc.build(story)
    return output_path
