"""
Interactive Data Visualization Components using Plotly.
Provides modern, responsive charts for candidate rankings, skill radar analysis,
score breakdowns, and ATS health gauges.
"""

from typing import List, Dict, Any
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def plot_leaderboard(results: List[Dict[str, Any]]) -> go.Figure:
    """Generate an interactive stacked horizontal bar chart of candidate scores and components."""
    names = [r["candidate_name"] for r in results][::-1]
    skill_scores = [r["scores_breakdown"]["skill_score"] for r in results][::-1]
    semantic_scores = [r["scores_breakdown"]["semantic_score"] for r in results][::-1]
    exp_scores = [r["scores_breakdown"]["exp_score"] for r in results][::-1]
    edu_scores = [r["scores_breakdown"]["edu_score"] for r in results][::-1]
    total_scores = [r["composite_score"] for r in results][::-1]

    fig = go.Figure()

    # We plot the composite scores with custom color mapping
    colors = []
    for s in total_scores:
        if s >= 80:
            colors.append("#10b981")  # Emerald green
        elif s >= 65:
            colors.append("#3b82f6")  # Blue
        elif s >= 50:
            colors.append("#f59e0b")  # Amber
        else:
            colors.append("#ef4444")  # Red

    fig.add_trace(go.Bar(
        y=names,
        x=total_scores,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(255, 255, 255, 0.4)', width=1)
        ),
        text=[f"<b>{s}%</b>" for s in total_scores],
        textposition='outside',
        customdata=list(zip(skill_scores, semantic_scores, exp_scores, edu_scores)),
        hovertemplate=(
            "<b>%{y}</b><br>" +
            "Composite Match: <b>%{x}%</b><br>" +
            "• Skill Coverage: %{customdata[0]}%<br>" +
            "• Semantic Similarity: %{customdata[1]}%<br>" +
            "• Experience Score: %{customdata[2]}%<br>" +
            "• Education Score: %{customdata[3]}%<extra></extra>"
        )
    ))

    fig.update_layout(
        title="<b>Candidate Screening Leaderboard</b>",
        xaxis=dict(title="Composite Relevance Score (%)", range=[0, 115]),
        yaxis=dict(title="", automargin=True),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=40, t=50, b=30),
        height=max(320, len(results) * 55),
        font=dict(family="sans-serif", size=13)
    )

    return fig


def plot_score_breakdown(results: List[Dict[str, Any]]) -> go.Figure:
    """Grouped bar chart showing multi-factor dimension scores across candidates."""
    names = [r["candidate_name"] for r in results]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Skill Coverage",
        x=names,
        y=[r["scores_breakdown"]["skill_score"] for r in results],
        marker_color="#6366f1"
    ))
    fig.add_trace(go.Bar(
        name="Semantic Similarity",
        x=names,
        y=[r["scores_breakdown"]["semantic_score"] for r in results],
        marker_color="#06b6d4"
    ))
    fig.add_trace(go.Bar(
        name="Experience Match",
        x=names,
        y=[r["scores_breakdown"]["exp_score"] for r in results],
        marker_color="#10b981"
    ))
    fig.add_trace(go.Bar(
        name="Education Match",
        x=names,
        y=[r["scores_breakdown"]["edu_score"] for r in results],
        marker_color="#f59e0b"
    ))

    fig.update_layout(
        title="<b>Multi-Factor Dimension Comparison</b>",
        barmode='group',
        yaxis=dict(title="Score (%)", range=[0, 105]),
        xaxis=dict(title=""),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=30),
        height=380
    )
    return fig


def plot_skill_radar(cat_breakdown: Dict[str, Dict[str, Any]]) -> go.Figure:
    """Radar (spider) chart comparing candidate skills against JD requirements by domain."""
    categories = []
    jd_counts = []
    matched_counts = []

    for cat, data in cat_breakdown.items():
        if cat != "Other / Custom" and (data["jd_count"] > 0 or data["cand_count"] > 0):
            categories.append(cat)
            jd_counts.append(data["jd_count"])
            matched_counts.append(data["matched_count"])

    if not categories:
        categories = ["Skills"]
        jd_counts = [1]
        matched_counts = [1]

    # Close the radar loop
    radar_cats = categories + [categories[0]]
    radar_jd = jd_counts + [jd_counts[0]]
    radar_matched = matched_counts + [matched_counts[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=radar_jd,
        theta=radar_cats,
        fill='toself',
        fillcolor='rgba(99, 102, 241, 0.15)',
        line=dict(color='#6366f1', dash='dash'),
        name='Job Requirements'
    ))
    fig.add_trace(go.Scatterpolar(
        r=radar_matched,
        theta=radar_cats,
        fill='toself',
        fillcolor='rgba(16, 185, 129, 0.35)',
        line=dict(color='#10b981', width=2),
        name='Candidate Matched'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, showticklabels=True),
            bgcolor="rgba(0,0,0,0)"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=30, b=50),
        height=380
    )
    return fig


def plot_score_gauge(score: float, title: str = "Relevance Score") -> go.Figure:
    """Plotly Indicator Gauge displaying individual score with graded zones."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"<b>{title}</b>", 'font': {'size': 16}},
        number={'suffix': "%", 'font': {'size': 28, 'color': '#1f2937'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': "#3b82f6", 'thickness': 0.28},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "#e5e7eb",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.2)'},
                {'range': [50, 65], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [65, 80], 'color': 'rgba(59, 130, 246, 0.2)'},
                {'range': [80, 100], 'color': 'rgba(16, 185, 129, 0.25)'}
            ],
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=220,
        margin=dict(l=20, r=20, t=40, b=10)
    )
    return fig


def plot_candidate_comparison(cand_a: Dict[str, Any], cand_b: Dict[str, Any]) -> go.Figure:
    """Generate a grouped bar chart directly comparing two candidates across all metrics."""
    metrics = ["Composite Score", "Skill Coverage", "Semantic Alignment", "Experience Match", "Education Rank"]
    val_a = [
        cand_a["composite_score"],
        cand_a["scores_breakdown"]["skill_score"],
        cand_a["scores_breakdown"]["semantic_score"],
        cand_a["scores_breakdown"]["exp_score"],
        cand_a["scores_breakdown"]["edu_score"]
    ]
    val_b = [
        cand_b["composite_score"],
        cand_b["scores_breakdown"]["skill_score"],
        cand_b["scores_breakdown"]["semantic_score"],
        cand_b["scores_breakdown"]["exp_score"],
        cand_b["scores_breakdown"]["edu_score"]
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=metrics,
        y=val_a,
        name=cand_a["candidate_name"],
        marker_color="#2563eb",
        text=[f"{v}%" for v in val_a],
        textposition="outside"
    ))
    fig.add_trace(go.Bar(
        x=metrics,
        y=val_b,
        name=cand_b["candidate_name"],
        marker_color="#10b981",
        text=[f"{v}%" for v in val_b],
        textposition="outside"
    ))

    fig.update_layout(
        title=f"<b>Head-to-Head Comparison: {cand_a['candidate_name']} vs {cand_b['candidate_name']}</b>",
        barmode="group",
        yaxis=dict(title="Score (%)", range=[0, 115]),
        xaxis=dict(title=""),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=30),
        height=380
    )
    return fig


