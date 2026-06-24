"""
app.py — Hotel Booking Cancellation Risk & Overbooking Strategy Dashboard
BAMD | IIM Calcutta

A Revenue-Manager-facing dashboard built on top of pipeline.py (the same
Streamlit-free analytics module used in the companion Colab notebook),
so every number here matches the notebook exactly.

Run locally with:  streamlit run app.py
Deploy on Streamlit Community Cloud by pointing it at this file in the
repo root (pipeline.py, data/, and models/ must ship alongside it).
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

import pipeline as pl

# =================================================================
# PAGE CONFIG & VISUAL IDENTITY
# =================================================================
st.set_page_config(
    page_title="Hotel Cancellation Risk & Overbooking Strategy",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY, AMBER, CORAL, SAGE, SLATE, GOLD = "#1B4965", "#D88C56", "#BC4749", "#5FA777", "#94A3B8", "#E3B23C"
HOTEL_COLORS = {"City Hotel": NAVY, "Resort Hotel": AMBER}
CANCEL_COLORS = {0: SAGE, 1: CORAL}
RISK_BAND_COLORS = {"Very Low": SAGE, "Low": "#9DBF8E", "Medium": GOLD, "High": AMBER, "Very High": CORAL}
RISK_BAND_ORDER = ["Very Low", "Low", "Medium", "High", "Very High"]

PLOTLY_TEMPLATE = go.layout.Template()
PLOTLY_TEMPLATE.layout = go.Layout(
    font=dict(family="Helvetica, Arial, sans-serif", color="#222", size=13),
    paper_bgcolor="white",
    plot_bgcolor="white",
    colorway=[NAVY, AMBER, CORAL, SAGE, SLATE, GOLD],
    title=dict(font=dict(size=17, color="#1A1A2E")),
    xaxis=dict(gridcolor="#EBEDF0", zerolinecolor="#D8DCE2", linecolor="#D8DCE2"),
    yaxis=dict(gridcolor="#EBEDF0", zerolinecolor="#D8DCE2", linecolor="#D8DCE2"),
    legend=dict(bgcolor="rgba(255,255,255,0.85)"),
    margin=dict(t=60, b=40, l=10, r=10),
)

st.markdown(f"""
<style>
    .block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1300px; }}
    [data-testid="stMetric"] {{
        background: #F7F9FB; border: 1px solid #E7EBEF; border-radius: 10px;
        padding: 14px 16px 10px 16px;
    }}
    [data-testid="stMetricLabel"] {{ color: #5B6472; font-weight: 600; }}
    h1, h2, h3 {{ color: #14213D; }}
    .insight-box {{
        background: linear-gradient(135deg, #F7F9FB 0%, #EFF3F6 100%);
        border-left: 4px solid {NAVY}; border-radius: 8px;
        padding: 14px 18px; margin: 10px 0 18px 0; font-size: 0.95rem; line-height: 1.55;
    }}
    .anomaly-box {{
        background: #FBF2F1; border-left: 4px solid {CORAL}; border-radius: 8px;
        padding: 14px 18px; margin: 10px 0 18px 0; font-size: 0.95rem; line-height: 1.55;
    }}
    .stTabs [data-baseweb="tab"] {{ font-size: 1rem; font-weight: 600; }}
</style>
""", unsafe_allow_html=True)


# =================================================================
# DATA / MODEL LOADING  (cached — same pipeline as the notebook)
# =================================================================
DATA_PATH = "data/hotel_bookings.csv"
MODELS_DIR = "models"


@st.cache_resource(show_spinner="Loading cancellation-risk models…")
def get_pipeline_state():
    """Loads pre-trained artifacts if they exist (fast path, used in
    deployment); otherwise runs the full pipeline from the raw CSV and
    saves artifacts for next time (first-run / local dev fallback)."""
    has_artifacts = os.path.isdir(MODELS_DIR) and any(
        f.startswith("model_") for f in os.listdir(MODELS_DIR)
    )
    if has_artifacts:
        try:
            return pl.load_artifacts(MODELS_DIR)
        except Exception:
            pass  # fall through to a full (re)run if artifacts are corrupt/incomplete

    state = pl.run_full_pipeline(DATA_PATH)
    try:
        pl.save_artifacts(state, MODELS_DIR)
    except Exception:
        pass  # read-only filesystem in some deployments -- not fatal
    return state


@st.cache_data(show_spinner=False)
def get_features_frame():
    """A lighter, EDA-friendly frame (post-cleaning, post-feature-engineering,
    pre train/test split) used by the Explore tab so filters can range over
    the entire dataset, not just the test period."""
    raw = pl.load_raw(DATA_PATH)
    clean, cleaning_report = pl.clean_data(raw)
    features = pl.engineer_features(clean)
    return features, cleaning_report


state = get_pipeline_state()
features, cleaning_report = get_features_frame()
test = state["test"]
train = state["train"]
models = state["models"]
best_model = state["best_model"]
probability_model = state["probability_model"]
test_probabilities = state["test_probabilities"]

risk_seg = pl.build_risk_segments(test, test_probabilities)
risk_profile = pl.profile_segments(risk_seg)

# =================================================================
# HEADER
# =================================================================
st.title("🏨 Hotel Booking Cancellation Risk & Overbooking Strategy")
st.caption("BAMD | IIM Calcutta  —  Revenue-Manager dashboard built on the same pipeline as the project notebook")

with st.sidebar:
    st.header("About this dashboard")
    st.markdown("""
This tool turns a booking-time cancellation-risk model into three concrete actions:

1. **Explore** what actually drives cancellations in this hotel group's own history.
2. **Score** today's bookings into risk tiers for retention outreach.
3. **Size** the overbooking buffer that maximises revenue without excessive walk-risk.

All numbers are computed by the same `pipeline.py` module used in the analysis notebook — nothing here is a separate, re-derived calculation.
""")
    st.divider()
    st.metric("Bookings analysed", f"{len(features):,}")
    st.metric("Best model (test ROC-AUC)", f"{pl.evaluate_model(best_model, test)['roc_auc']:.3f}",
              help="Gradient Boosting, evaluated on bookings made after the training cut-off date — see the notebook's Section 15 for why a time-based split matters here.")
    st.divider()
    st.caption("Dataset: Hotel Booking Demand (Kaggle / Antonio, Almeida & Nunes, 2019)")

tab_overview, tab_explore, tab_models, tab_risk, tab_overbook, tab_notes = st.tabs(
    ["🏨 Overview", "🔍 Explore the Data", "🤖 Model Performance",
     "🎯 Risk & Retention", "📊 Overbooking Simulator", "📝 Methodology & Notes"]
)

# =================================================================
# TAB 1 — OVERVIEW
# =================================================================
with tab_overview:
    st.subheader("The business problem, in one line")
    st.markdown("""
Booking cancellations are the largest *controllable* leakage in hotel revenue. Treat every booking as firm,
and cancelled rooms sit empty. Overbook to compensate without a real model behind it, and you risk walking
arriving guests to a competitor — costly in cash and in reputation. This dashboard replaces the guess with a
booking-time cancellation-risk model, trained and validated end-to-end in the companion Colab notebook.
""")

    overall_rate = features.is_canceled.mean()
    city_rate = features.loc[features.hotel == "City Hotel", "is_canceled"].mean()
    resort_rate = features.loc[features.hotel == "Resort Hotel", "is_canceled"].mean()
    gb_res = pl.evaluate_model(best_model, test)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall cancellation rate", f"{overall_rate:.1%}")
    c2.metric("City Hotel rate", f"{city_rate:.1%}")
    c3.metric("Resort Hotel rate", f"{resort_rate:.1%}")
    c4.metric("Model ROC-AUC", f"{gb_res['roc_auc']:.3f}", help="Gradient Boosting, out-of-time test set")

    st.markdown("#### Answering the project's eight questions")
    qa_df = pd.DataFrame([
        ("Q1", "Can we predict cancellation at booking time?", f"Yes — ROC-AUC {gb_res['roc_auc']:.3f} on an out-of-time test set, using only booking-time information."),
        ("Q2", "Does longer lead time raise risk?", "Yes, sharply — risk climbs almost monotonically from ~8% (0–7 days) to over 40% (365+ days)."),
        ("Q3", "Which channel/segment cancels most?", "Online TA cancels at the highest rate (~35%) among sizeable segments; Groups (~27%) follows. Direct/Corporate are most reliable (~12–15%)."),
        ("Q4", "Do deposit types behave as expected?", "No — Non-Refund bookings cancel ~95% of the time. Investigated in Explore tab: a narrow City-Hotel/Groups/Portugal booking cluster, not deposits generically."),
        ("Q5", "Do special requests predict show-up?", "Yes — more requests, steadily lower cancellation risk. A free, model-free booking-quality signal."),
        ("Q6", "Do repeat guests / past cancellers differ?", "Yes, strongly — prior cancellers cancel again ~68% of the time; repeat guests cancel ~3.5x less often than first-timers."),
        ("Q7", "City vs Resort cancellation behaviour?", f"City Hotel cancels more often ({city_rate:.0%} vs {resort_rate:.0%}) — see the Overbooking tab for property-specific buffers."),
        ("Q8", "What overbooking buffer maximises revenue?", "Computed live in the Overbooking Simulator tab, under your own capacity/cost assumptions."),
    ], columns=["#", "Question", "Answer"])
    st.dataframe(qa_df, hide_index=True, width='stretch',
                 column_config={"#": st.column_config.TextColumn(width="small")})

    st.markdown("#### Top 3 policy levers")
    st.markdown(f"""
<div class="insight-box">
<b>1. Scale deposit requirements with lead time and channel</b>, not a blanket rule — long-lead, Online-TA and Group bookings carry most of the controllable risk.<br><br>
<b>2. Flag bookings for retention outreach far below the textbook 0.5 probability cutoff</b> — see the Risk & Retention tab for a cost-sensitive threshold tuned to your own numbers.<br><br>
<b>3. Run separate overbooking buffers per property</b> — a one-size-fits-all buffer is too cautious for the City Hotel and too aggressive for the Resort.
</div>
""", unsafe_allow_html=True)

# =================================================================
# TAB 2 — EXPLORE THE DATA
# =================================================================
with tab_explore:
    st.subheader("Explore cancellation patterns")
    st.caption("Filters apply to the full cleaned dataset (all properties, all dates) — not just the model's test period.")

    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        f_hotel = st.multiselect("Hotel", sorted(features.hotel.unique()), default=sorted(features.hotel.unique()))
    with fc2:
        f_segment = st.multiselect("Market segment", sorted(features.market_segment.unique()),
                                    default=sorted(features.market_segment.unique()))
    with fc3:
        f_deposit = st.multiselect("Deposit type", sorted(features.deposit_type.unique()),
                                    default=sorted(features.deposit_type.unique()))
    with fc4:
        f_lead = st.slider("Lead time (days)", 0, int(features.lead_time.max()), (0, int(features.lead_time.max())))

    fdf = features[
        features.hotel.isin(f_hotel) & features.market_segment.isin(f_segment) &
        features.deposit_type.isin(f_deposit) &
        features.lead_time.between(f_lead[0], f_lead[1])
    ]

    if len(fdf) == 0:
        st.warning("No bookings match these filters — widen your selection.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Bookings in selection", f"{len(fdf):,}")
        m2.metric("Cancellation rate", f"{fdf.is_canceled.mean():.1%}")
        m3.metric("Revenue at risk (cancelled)", f"€{fdf.loc[fdf.is_canceled==1,'expected_revenue'].sum():,.0f}")

        colA, colB = st.columns(2)
        with colA:
            bins = [-1, 7, 30, 90, 180, 365, 10000]
            labels = ["0-7d", "8-30d", "31-90d", "91-180d", "181-365d", "365d+"]
            fdf2 = fdf.assign(lead_bucket=pd.cut(fdf.lead_time, bins=bins, labels=labels))
            lead_rate = fdf2.groupby("lead_bucket", observed=True).is_canceled.mean().reindex(labels)
            fig = px.line(x=lead_rate.index, y=lead_rate.values, markers=True,
                          labels={"x": "Lead time", "y": "Cancellation rate"},
                          title="Cancellation Rate by Lead Time")
            fig.update_traces(line_color=CORAL, marker=dict(size=9))
            fig.update_yaxes(tickformat=".0%")
            fig.update_layout(template=PLOTLY_TEMPLATE)
            st.plotly_chart(fig, width='stretch')

        with colB:
            seg_rate = fdf.groupby("market_segment").is_canceled.mean().sort_values(ascending=False)
            seg_vol = fdf.market_segment.value_counts().reindex(seg_rate.index)
            fig = px.bar(x=seg_rate.values, y=seg_rate.index, orientation="h",
                         labels={"x": "Cancellation rate", "y": ""}, title="Cancellation Rate by Market Segment",
                         text=[f"n={v:,}" for v in seg_vol.values])
            fig.update_traces(marker_color=NAVY, textposition="outside")
            fig.update_xaxes(tickformat=".0%")
            fig.update_layout(template=PLOTLY_TEMPLATE, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, width='stretch')

        colC, colD = st.columns(2)
        with colC:
            dep_rate = fdf.groupby("deposit_type").is_canceled.mean()
            fig = px.bar(x=dep_rate.index, y=dep_rate.values, labels={"x": "", "y": "Cancellation rate"},
                         title="Cancellation Rate by Deposit Type", text=[f"{v:.0%}" for v in dep_rate.values])
            fig.update_traces(marker_color=[CORAL if v > overall_rate else SAGE for v in dep_rate.values],
                               textposition="outside")
            fig.update_yaxes(tickformat=".0%")
            fig.update_layout(template=PLOTLY_TEMPLATE)
            st.plotly_chart(fig, width='stretch')

        with colD:
            req_rate = fdf.groupby("total_of_special_requests").is_canceled.mean()
            fig = px.bar(x=req_rate.index.astype(str), y=req_rate.values,
                         labels={"x": "Number of special requests", "y": "Cancellation rate"},
                         title="Cancellation Rate by Special Requests")
            fig.update_traces(marker_color=NAVY)
            fig.update_yaxes(tickformat=".0%")
            fig.update_layout(template=PLOTLY_TEMPLATE)
            st.plotly_chart(fig, width='stretch')

        monthly = fdf.set_index("arrival_date").sort_index().resample("ME")["is_canceled"].mean()
        fig = px.line(x=monthly.index, y=monthly.values, markers=True,
                      labels={"x": "Arrival month", "y": "Cancellation rate"},
                      title="Cancellation Rate Over Time")
        fig.update_traces(line_color=CORAL, marker=dict(size=6))
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, width='stretch')

        if "Non Refund" in f_deposit:
            nr = fdf[fdf.deposit_type == "Non Refund"]
            if len(nr) > 0:
                st.markdown(f"""
<div class="anomaly-box">
<b>⚠️ About the Non-Refund deposit anomaly:</b> in your current selection, Non-Refund bookings cancel at
<b>{nr.is_canceled.mean():.0%}</b> — counter-intuitively high for a deposit that should signal commitment.
Investigation in the project notebook traces this to a narrow, concentrated booking pattern: ~81% City Hotel,
~91% Groups/Offline-TA, ~95% from Portugal, with a median lead time nearly 4x the dataset average — almost
certainly provisional bulk group reservations from domestic tour operators, not a generic statement about
how deposits affect guest behaviour. Treat this feature as a real but narrow signal, not a universal rule.
</div>
""", unsafe_allow_html=True)

# =================================================================
# TAB 3 — MODEL PERFORMANCE
# =================================================================
with tab_models:
    st.subheader("Model comparison")
    st.caption("All models trained on bookings made before 1 Feb 2017, evaluated on bookings made on/after that date — a genuine out-of-time test, not a random shuffle.")

    comparison_rows = []
    roc_data, pr_data = {}, {}
    for name, pipe in models.items():
        res = pl.evaluate_model(pipe, test)
        comparison_rows.append(dict(Model=name, **{k: res[k] for k in
                                ["roc_auc", "pr_auc", "recall", "precision", "f1", "brier"]}))
        roc_data[name] = pl.get_roc_curve(res["y_true"], res["y_prob"])
        pr_data[name] = pl.get_pr_curve(res["y_true"], res["y_prob"])
    comparison_df = pd.DataFrame(comparison_rows).rename(columns={
        "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "recall": "Recall",
        "precision": "Precision", "f1": "F1", "brier": "Brier (lower better)"})
    st.dataframe(
        comparison_df.style.format({c: "{:.3f}" for c in comparison_df.columns if c != "Model"})
        .background_gradient(subset=["ROC-AUC", "F1"], cmap="Greens"),
        hide_index=True, width='stretch',
    )

    model_colors = {"Logistic Regression": SLATE, "Decision Tree": GOLD, "Random Forest": NAVY, "Gradient Boosting": CORAL}
    colA, colB = st.columns(2)
    with colA:
        fig = go.Figure()
        for name, (fpr, tpr, _) in roc_data.items():
            auc = comparison_df.loc[comparison_df.Model == name, "ROC-AUC"].iloc[0]
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} ({auc:.3f})",
                                      line=dict(color=model_colors[name], width=2.5)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color="gray", dash="dash"), showlegend=False))
        fig.update_layout(title="ROC Curves", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                           template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, width='stretch')
    with colB:
        fig = go.Figure()
        for name, (prec, rec, _) in pr_data.items():
            ap = comparison_df.loc[comparison_df.Model == name, "PR-AUC"].iloc[0]
            fig.add_trace(go.Scatter(x=rec, y=prec, name=f"{name} ({ap:.3f})",
                                      line=dict(color=model_colors[name], width=2.5)))
        fig.add_hline(y=test.is_canceled.mean(), line_dash="dash", line_color="gray")
        fig.update_layout(title="Precision-Recall Curves", xaxis_title="Recall", yaxis_title="Precision",
                           template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, width='stretch')

    st.markdown("#### Feature importance")
    fi_model = st.selectbox("Model", ["Random Forest", "Gradient Boosting"], index=1,
                             help="Logistic Regression's drivers are shown as odds ratios below instead — coefficients aren't directly comparable to tree importances.")
    if fi_model == "Random Forest":
        imp = pl.tree_feature_importance(models["Random Forest"])
        imp_label = "Gini importance"
    else:
        with st.spinner("Computing permutation importance…"):
            imp = pl.tree_feature_importance(
                models["Gradient Boosting"], X=test.drop(columns=["is_canceled"]), y=test["is_canceled"],
                n_repeats=4, sample_size=6000,
            )
        imp_label = "Permutation importance (ROC-AUC drop)"
    fig = px.bar(imp.head(12).sort_values("importance"), x="importance", y="feature", orientation="h",
                 title=f"{fi_model} — {imp_label}", labels={"importance": imp_label, "feature": ""})
    fig.update_traces(marker_color=NAVY if fi_model == "Random Forest" else CORAL)
    fig.update_layout(template=PLOTLY_TEMPLATE)
    st.plotly_chart(fig, width='stretch')

    with st.expander("Logistic Regression — plain-English odds ratios"):
        odds = pl.odds_ratios_from_logreg(models["Logistic Regression"])
        st.caption("Odds ratio > 1 raises cancellation odds; < 1 lowers them. Sorted by overall strength of effect.")
        st.dataframe(odds.head(15).style.format({"coefficient": "{:.3f}", "odds_ratio": "{:.2f}"}),
                     hide_index=True, width='stretch')

    st.markdown("#### Two findings worth knowing about before trusting these numbers")
    nc1, nc2 = st.columns(2)
    with nc1:
        st.markdown("""
**SMOTE vs. class weighting:** the project plan calls for testing SMOTE oversampling against `class_weight="balanced"`.
We tested both on Random Forest — SMOTE cost roughly 15x more compute for *worse* recall. We use class weighting
throughout; see the notebook (Section 16) for the side-by-side numbers.
""")
    with nc2:
        st.markdown("""
**Probability calibration:** `class_weight="balanced"` is excellent for ranking risk but inflates the actual
probability values (mean predicted ~0.38 vs. an actual rate of ~0.28 in one check). The Risk & Retention and
Overbooking tabs use a separate, unweighted **probability model** instead — see notebook Section 17.
""")

# =================================================================
# TAB 4 — RISK SEGMENTATION & RETENTION
# =================================================================
with tab_risk:
    st.subheader("Risk tiers")
    st.caption("Every test-set booking scored by the probability model (Section 17 of the notebook explains why this is a *different* model from the headline classifiers) and bucketed into five tiers.")

    rp = risk_profile.set_index("risk_band").reindex(RISK_BAND_ORDER).reset_index()
    colA, colB = st.columns(2)
    with colA:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=rp.risk_band, y=rp.actual_cancel_rate,
                              marker_color=[RISK_BAND_COLORS[b] for b in rp.risk_band], name="Actual cancel rate"))
        fig.add_trace(go.Scatter(x=rp.risk_band, y=rp.avg_predicted_prob, mode="lines+markers",
                                  marker=dict(color="black", size=9, symbol="diamond"),
                                  line=dict(color="black"), name="Avg. predicted probability"))
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(title="Risk Tier vs Actual Cancellation Rate", template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, width='stretch')
    with colB:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=rp.risk_band, y=rp.bookings, marker_color=[RISK_BAND_COLORS[b] for b in rp.risk_band],
                              name="Bookings", yaxis="y1"))
        fig.add_trace(go.Scatter(x=rp.risk_band, y=rp.revenue_at_risk, mode="lines+markers",
                                  marker=dict(color=NAVY, size=8), line=dict(color=NAVY),
                                  name="Revenue at risk (€)", yaxis="y2"))
        fig.update_layout(title="Bookings per Tier vs Revenue at Risk", template=PLOTLY_TEMPLATE,
                           yaxis=dict(title="Bookings"), yaxis2=dict(title="Revenue at risk (€)", overlaying="y", side="right"))
        st.plotly_chart(fig, width='stretch')

    rp_display = rp.copy()
    rp_display["revenue_at_risk"] = rp_display["revenue_at_risk"].map(lambda x: f"€{x:,.0f}")
    for c in ["actual_cancel_rate", "avg_predicted_prob", "pct_no_deposit", "pct_online_ta"]:
        rp_display[c] = rp_display[c].map(lambda x: f"{x:.1%}")
    st.dataframe(rp_display.rename(columns={
        "risk_band": "Risk Band", "bookings": "Bookings", "actual_cancel_rate": "Actual Cancel Rate",
        "avg_predicted_prob": "Avg. Predicted Prob.", "avg_lead_time": "Avg. Lead Time (days)",
        "avg_adr": "Avg. ADR (€)", "avg_special_requests": "Avg. Special Requests",
        "revenue_at_risk": "Revenue at Risk", "pct_no_deposit": "% No Deposit", "pct_online_ta": "% Online TA",
    }), hide_index=True, width='stretch')

    st.divider()
    st.subheader("Who should get a retention call today?")
    st.caption("A different decision from overbooking: which *individual* bookings are worth a proactive call, confirmation email, or small incentive — before they cancel.")

    rc1, rc2 = st.columns(2)
    with rc1:
        cost_fn = st.slider("Cost of a MISSED cancellation (€)", 10, 500, 120, step=10,
                             help="Revenue lost when a booking that was going to cancel wasn't flagged for outreach.")
    with rc2:
        cost_fp = st.slider("Cost of an UNNECESSARY call (€)", 1, 100, 4, step=1,
                             help="Agent time / discount cost on a booking that would have honoured anyway.")

    threshold_costs = pl.cost_sensitive_threshold(test.is_canceled, test_probabilities, cost_fn=cost_fn, cost_fp=cost_fp)
    best_row = threshold_costs.loc[threshold_costs.total_cost.idxmin()]
    default_row = threshold_costs.loc[(threshold_costs.threshold - 0.5).abs().idxmin()]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=threshold_costs.threshold, y=threshold_costs.total_cost,
                              line=dict(color=NAVY, width=2.5), mode="lines+markers", marker=dict(size=5)))
    fig.add_vline(x=best_row.threshold, line_dash="dash", line_color=CORAL,
                  annotation_text=f"Optimal = {best_row.threshold:.2f}")
    fig.add_vline(x=0.5, line_dash="dot", line_color=SLATE, annotation_text="Default = 0.50")
    fig.update_layout(title="Total Expected Cost vs. Decision Threshold", template=PLOTLY_TEMPLATE,
                       xaxis_title="Probability threshold for flagging a retention call", yaxis_title="Total expected cost (€)")
    st.plotly_chart(fig, width='stretch')

    savings = default_row.total_cost - best_row.total_cost
    st.success(
        f"**Cost-minimising threshold: {best_row.threshold:.2f}** "
        f"(total expected cost €{best_row.total_cost:,.0f}, vs €{default_row.total_cost:,.0f} at the default 0.50 cutoff "
        f"— a saving of €{savings:,.0f} on this test period alone)."
    )

    st.divider()
    st.subheader("Score a hypothetical booking")
    st.caption("A quick what-if tool: describe a booking and see its predicted cancellation risk and tier.")

    bc1, bc2, bc3, bc4 = st.columns(4)
    with bc1:
        b_hotel = st.selectbox("Hotel", ["City Hotel", "Resort Hotel"])
        b_lead = st.number_input("Lead time (days)", 0, 700, 60)
        b_nights = st.number_input("Total nights", 1, 30, 3)
    with bc2:
        b_segment = st.selectbox("Market segment", sorted(features.market_segment.unique()), index=0)
        b_channel = st.selectbox("Distribution channel", sorted(features.distribution_channel.unique()))
        b_deposit = st.selectbox("Deposit type", sorted(features.deposit_type.unique()))
    with bc3:
        b_country = st.selectbox("Country", sorted(features.country_grouped.unique()))
        b_customer = st.selectbox("Customer type", sorted(features.customer_type.unique()))
        b_adr = st.number_input("ADR (€/night)", 0, 1000, 110)
    with bc4:
        b_requests = st.slider("Special requests", 0, 5, 0)
        b_prior_cancel = st.number_input("Prior cancellations", 0, 20, 0)
        b_parking = st.checkbox("Requested parking")

    if st.button("Score this booking", type="primary"):
        hypothetical = pd.DataFrame([{
            "lead_time": b_lead, "arrival_date_week_number": 25, "stays_in_weekend_nights": b_nights // 3,
            "stays_in_week_nights": b_nights - b_nights // 3, "adults": 2, "children": 0, "babies": 0,
            "total_guests": 2, "is_repeated_guest": 0, "previous_cancellations": b_prior_cancel,
            "previous_bookings_not_canceled": 0, "booking_changes": 0, "agent_lead_time_dev": 0,
            "days_in_waiting_list": 0, "adr": b_adr, "required_car_parking_spaces": int(b_parking),
            "total_of_special_requests": b_requests, "booked_via_company": 0, "arrival_month_num": 6,
            "hotel": b_hotel, "meal": "BB", "country_grouped": b_country, "market_segment": b_segment,
            "distribution_channel": b_channel, "reserved_room_type": "A", "deposit_type": b_deposit,
            "customer_type": b_customer, "season": "Summer", "arrival_day_of_week": "Friday",
            "total_nights": b_nights,
        }])
        pred_prob = probability_model.predict_proba(hypothetical)[0, 1]
        band = pl.assign_risk_band(pred_prob)
        rc1, rc2 = st.columns([1, 2])
        with rc1:
            st.metric("Predicted cancellation probability", f"{pred_prob:.1%}")
            st.markdown(f"**Risk tier:** <span style='color:{RISK_BAND_COLORS[band]}; font-weight:700; font-size:1.2rem'>{band}</span>",
                        unsafe_allow_html=True)
        with rc2:
            recommendation = {
                "Very Low": "No action needed — this booking profile is very likely to be honoured.",
                "Low": "No action needed; revisit only if other risk factors emerge later.",
                "Medium": "Optional light-touch confirmation email closer to arrival.",
                "High": "Recommend a proactive retention call or incentive offer.",
                "Very High": "Strong candidate for immediate outreach and/or a tightened deposit term.",
            }[band]
            st.info(recommendation)

# =================================================================
# TAB 5 — OVERBOOKING SIMULATOR
# =================================================================
with tab_overbook:
    st.subheader("How many extra bookings should we accept tonight?")
    st.caption("Monte-Carlo simulation using the calibrated probability model's actual cancellation probabilities — see notebook Section 20 for the full method.")

    oc1, oc2, oc3, oc4 = st.columns(4)
    with oc1:
        sim_hotel = st.selectbox("Property", ["City Hotel", "Resort Hotel", "Both (pooled)"])
    with oc2:
        sim_capacity = st.number_input("Room capacity", 10, 1000, 100, step=10)
    with oc3:
        sim_walk_cost = st.slider("Walk cost (× nightly rate)", 1.0, 10.0, 4.0, step=0.5,
                                   help="Covers compensation, rebooking at a competitor, and the reputational hit of walking a guest.")
    with oc4:
        sim_max_buffer = st.number_input("Max buffer to test", 10, 300, 110, step=10)

    if sim_hotel == "Both (pooled)":
        sim_pool = risk_seg
    else:
        sim_pool = risk_seg[risk_seg.hotel == sim_hotel]

    run_sim = st.button("🎲 Run simulation", type="primary")

    if run_sim or "sim_result" in st.session_state:
        if run_sim:
            buffer_range = np.arange(0, sim_max_buffer + 1, max(5, sim_max_buffer // 20))
            with st.spinner("Running Monte-Carlo simulation…"):
                sim = pl.simulate_overbooking(
                    sim_pool.adr.values, sim_pool.cancel_probability.values,
                    capacity=sim_capacity, buffer_sizes=buffer_range,
                    walk_cost_multiplier=sim_walk_cost, n_trials=400,
                )
            st.session_state["sim_result"] = sim
            st.session_state["sim_meta"] = (sim_hotel, sim_capacity, sim_walk_cost)
        else:
            sim = st.session_state["sim_result"]
            sim_hotel, sim_capacity, sim_walk_cost = st.session_state["sim_meta"]

        best = sim.loc[sim.expected_net_revenue.idxmax()]
        baseline = sim.loc[sim.buffer == 0, "expected_net_revenue"].iloc[0]
        uplift = best.expected_net_revenue / baseline - 1 if baseline != 0 else float("nan")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Optimal buffer", f"+{int(best.buffer)} rooms")
        m2.metric("Expected net revenue", f"€{best.expected_net_revenue:,.0f}")
        m3.metric("Revenue uplift vs. no overbooking", f"{uplift:.1%}")
        m4.metric("Walk-risk at optimal buffer", f"{best.prob_any_walk:.1%}",
                  help="Probability of walking at least one guest, at the revenue-maximising buffer.")

        colA, colB = st.columns(2)
        with colA:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sim.buffer, y=sim.expected_net_revenue, mode="lines+markers",
                                      line=dict(color=NAVY, width=2.5), marker=dict(size=6),
                                      fill="tozeroy", fillcolor="rgba(27,73,101,0.08)"))
            fig.add_vline(x=best.buffer, line_dash="dash", line_color=CORAL,
                          annotation_text=f"Optimal = +{int(best.buffer)}")
            fig.add_hline(y=0, line_color="black", line_width=1)
            fig.update_layout(title=f"{sim_hotel} — Expected Net Revenue vs Buffer", template=PLOTLY_TEMPLATE,
                               xaxis_title=f"Buffer (rooms above {sim_capacity}-room capacity)", yaxis_title="Expected net revenue (€)")
            st.plotly_chart(fig, width='stretch')
        with colB:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sim.buffer, y=sim.prob_any_walk, mode="lines+markers",
                                      line=dict(color=CORAL, width=2.5), marker=dict(size=6)))
            fig.add_hline(y=0.05, line_dash="dot", line_color=SLATE, annotation_text="5% reference line")
            fig.add_vline(x=best.buffer, line_dash="dash", line_color=NAVY)
            fig.update_yaxes(tickformat=".0%")
            fig.update_layout(title="Walk-Risk vs Buffer", template=PLOTLY_TEMPLATE,
                               xaxis_title=f"Buffer (rooms above {sim_capacity}-room capacity)", yaxis_title="P(walk at least one guest)")
            st.plotly_chart(fig, width='stretch')

        st.markdown(f"""
<div class="insight-box">
At <b>+{int(best.buffer)} rooms</b> above a {sim_capacity}-room capacity, the simulation expects
<b>{best.expected_walked_guests:.2f}</b> guests walked on average and <b>{best.expected_empty_rooms:.1f}</b>
empty rooms — versus <b>{sim.loc[sim.buffer==0,'expected_empty_rooms'].iloc[0]:.1f}</b> empty rooms today with
no overbooking at all. If walking even one guest occasionally is unacceptable for brand reasons, look one or
two buffer steps to the left of the revenue-maximising point in the left-hand chart — a small, deliberate
revenue trade-off for materially lower walk-risk.
</div>
""", unsafe_allow_html=True)

        with st.expander("See the full simulation table"):
            st.dataframe(sim.style.format({
                "expected_net_revenue": "€{:,.0f}", "std_net_revenue": "€{:,.0f}",
                "expected_walked_guests": "{:.2f}", "expected_empty_rooms": "{:.1f}",
                "p95_walked_guests": "{:.0f}", "prob_any_walk": "{:.1%}",
            }), hide_index=True, width='stretch')
    else:
        st.info("Set your assumptions above and click **Run simulation** to see the recommended buffer.")

# =================================================================
# TAB 6 — METHODOLOGY & NOTES
# =================================================================
with tab_notes:
    st.subheader("Data cleaning")
    st.markdown(f"""
- **Raw rows:** {cleaning_report.raw_rows:,} → **Modeling rows:** {cleaning_report.final_rows:,}
  ({cleaning_report.raw_rows - cleaning_report.final_rows:,} removed, mostly duplicate PMS export rows)
""")
    for note in cleaning_report.notes:
        st.markdown(f"- {note}")

    st.subheader("Leakage audit")
    st.markdown("""
Three columns are **permanently excluded** from every model because they are not known at the moment a booking
is made:
""")
    for c in pl.LEAKAGE_COLS:
        st.markdown(f"- `{c}`")
    st.markdown("""
A throwaway model trained *with* `reservation_status` reaches a perfect ROC-AUC of 1.0 on this dataset — not
because it found a brilliant signal, but because that column **is** the answer in disguise. See notebook
Section 5 for the full demonstration.
""")

    st.subheader("Why a time-based train/test split")
    st.markdown("""
Every model here is trained on bookings made before **1 February 2017** and evaluated only on bookings made
on/after that date. A random shuffle would let near-duplicate bookings from the same agent land on both sides
of the split, inflating reported accuracy in a way that wouldn't survive contact with a real, future booking.
Published benchmarks on this exact dataset often report ROC-AUC in the 0.92–0.97 range; those are typically
produced with a random split on data that still contains duplicate rows. Our honestly-lower 0.836 reflects how
well the model predicts cancellations it has genuinely never seen — the only way this would actually be used.
""")

    st.subheader("Two methodology choices, tested and documented")
    nc1, nc2 = st.columns(2)
    with nc1:
        st.markdown("""
**Class imbalance: `class_weight="balanced"` over SMOTE.** We tested SMOTE oversampling on Random Forest —
it cost ~15x more compute for *worse* recall than simple class weighting. With ~70K real training rows and a
72/28 split (not a severe imbalance), manufacturing synthetic minority-class points didn't help here.
""")
    with nc2:
        st.markdown("""
**A separate, unweighted "probability model" for simulation.** `class_weight="balanced"` ranks bookings by
risk excellently but inflates the actual probability values. Risk Segmentation and the Overbooking Simulator
use a separate, unweighted twin of the same architecture instead — its probabilities are well-calibrated,
confirmed with a 10-bin reliability check in the notebook (Section 17).
""")

    st.subheader("The Non-Refund deposit anomaly")
    st.markdown("""
Non-Refund bookings cancel at nearly 95% — backwards from how a deposit should work. Investigation traces this
to a narrow cluster: ~81% City Hotel, ~91% Groups/Offline-TA, ~95% Portugal, with a median lead time nearly 4x
the dataset average — almost certainly provisional bulk group reservations from domestic tour operators.
A second layer: over 92% of the *raw* Non-Refund rows were exact duplicates (versus ~27% for the dataset
overall), meaning a lot of this anomaly's apparent scale in casual analyses of this public dataset is a
data-export artifact, not 14,000 distinct decisions. We keep the feature — it's real and informative — but
flag it as concentrated and narrow rather than a universal statement about deposit psychology.
""")

    st.subheader("Known limitations")
    st.markdown("""
- The dataset doesn't record each property's actual room **capacity** — the Overbooking Simulator's capacity
  is whatever you enter; it isn't derived from the data.
- The walk-cost multiplier is an assumption standing in for compensation, rebooking, and reputational cost,
  none of which this dataset measures directly.
- `agent` and `company` are anonymised IDs with no descriptive metadata; engineered features on them
  (e.g. lead-time deviation) are necessarily behavioural rather than using real agent characteristics.
- The data covers 2015–2017 at two specific Portuguese properties. The *qualitative* drivers (lead time,
  deposit type, prior history, special requests) are intuitive enough to generalise; the exact thresholds and
  buffer sizes are specific to this hotel group's historical mix and should be re-validated on a new
  property's own data before being applied directly.
""")

    st.caption("Built with pipeline.py — the same module powering Project1_Hotel_Cancellation_Risk.ipynb. "
               "Re-running the notebook's final cell regenerates the artifacts this dashboard loads.")
