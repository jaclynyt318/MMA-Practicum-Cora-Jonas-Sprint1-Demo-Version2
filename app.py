
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

RAW_DIR = "data/raw"
DERIVED_DIR = "data/derived"

RAW_FILES = {
    "Accounts": "ravenstack_accounts.csv",
    "Subscriptions": "ravenstack_subscriptions.csv",
    "Feature Usage": "ravenstack_feature_usage.csv",
    "Support Tickets": "ravenstack_support_tickets.csv",
    "Churn Events": "ravenstack_churn_events.csv",
}

DERIVED_FILES = {
    "KPIs Snapshot": "kpis_snapshot.csv",
    "Signups by Month": "signups_by_month.csv",
    "MRR by Month": "mrr_by_month.csv",
    "MRR by Plan by Month": "mrr_by_plan_by_month.csv",
    "Plan Changes by Month": "plan_changes_by_month.csv",
    "Billing Split": "billing_split.csv",
    "Tickets by Month": "tickets_by_month.csv",
    "Ticket Priority Dist": "ticket_priority_dist.csv",
    "Satisfaction Coverage": "satisfaction_coverage.csv",
    "Churn Events by Month": "churn_events_by_month.csv",
    "Churn Reasons": "churn_reasons.csv",
    "Reactivation Summary": "reactivation_summary.csv",
    "Usage by Month": "usage_by_month.csv",
    "Top Features": "top_features.csv",
}

# ----------------------------
# Page config + global styling
# ----------------------------
st.set_page_config(
    page_title="EDA Dashboards",
    page_icon="📊",
    layout="wide",
)

# TailAdmin-ish styling.
# Key fix: Use Streamlit bordered containers instead of opening/closing HTML divs across multiple markdown calls.
CSS = """
<style>
/* Give breathing room so titles are not hidden under Streamlit's top bar */
.block-container { padding-top: 3.2rem; padding-bottom: 2.2rem; max-width: 1400px; }

h1, h2, h3 { letter-spacing: -0.02em; }

section[data-testid="stSidebar"] {
  background: #f8fafc;
  border-right: 1px solid #e5e7eb;
}

/* Bordered containers (Streamlit) */
div[data-testid="stVerticalBlockBorderWrapper"]{
  border-radius: 14px !important;
  border: 1px solid #e5e7eb !important;
  background: #ffffff !important;
  box-shadow: 0 1px 0 rgba(0,0,0,0.02);
}

/* KPI cards (custom HTML) */
.rs-card {
  border: 1px solid #e5e7eb;
  background: #ffffff;
  border-radius: 14px;
  padding: 14px 16px;
  box-shadow: 0 1px 0 rgba(0,0,0,0.02);
}
.rs-card-title {
  font-size: 12px;
  color: rgba(0,0,0,0.55);
  margin-bottom: 8px;
  font-weight: 600;
}
.rs-card-value {
  font-size: 26px;
  font-weight: 800;
  line-height: 1.1;
}
.rs-card-sub {
  margin-top: 8px;
  font-size: 12px;
  color: rgba(0,0,0,0.55);
}
.rs-pill {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 700;
  border: 1px solid #e5e7eb;
  background: #f9fafb;
}
.rs-pill-green { border-color: rgba(34,197,94,0.25); background: rgba(34,197,94,0.08); color: #166534; }
.rs-pill-red { border-color: rgba(239,68,68,0.25); background: rgba(239,68,68,0.08); color: #7f1d1d; }
.rs-pill-blue { border-color: rgba(59,130,246,0.25); background: rgba(59,130,246,0.08); color: #1e3a8a; }

/* Notes */
.rs-note {
  border: 1px solid #e5e7eb;
  background: #f9fafb;
  border-radius: 12px;
  padding: 12px 14px;
  font-size: 13px;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ----------------------------
# Helpers
# ----------------------------
def _exists(path: str) -> bool:
    return os.path.exists(path) and os.path.isfile(path)

def _raw_path(filename: str) -> str:
    return os.path.join(RAW_DIR, filename)

def _derived_path(filename: str) -> str:
    return os.path.join(DERIVED_DIR, filename)

@st.cache_data(show_spinner=False)
def load_raw(filename: str) -> pd.DataFrame:
    return pd.read_csv(_raw_path(filename))

@st.cache_data(show_spinner=False)
def load_derived(filename: str) -> pd.DataFrame:
    return pd.read_csv(_derived_path(filename))

def fmt_int(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{int(x):,}"

def fmt_money(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"${float(x):,.0f}"

def rs_card(title: str, value: str, sub: str = "", pill_text: str = "", pill_kind: str = "blue"):
    pill_cls = {"green": "rs-pill-green", "red": "rs-pill-red", "blue": "rs-pill-blue"}.get(pill_kind, "rs-pill-blue")
    pill_html = f'<span class="rs-pill {pill_cls}">{pill_text}</span>' if pill_text else ""
    st.markdown(
        f"""
        <div class="rs-card">
          <div class="rs-card-title">{title} {pill_html}</div>
          <div class="rs-card-value">{value}</div>
          <div class="rs-card-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def data_sources_note(sources, definitions=""):
    txt = f"Data sources: {', '.join(sources)}"
    if definitions:
        txt += f"<br/><b>Definitions:</b> {definitions}"
    st.markdown(f'<div class="rs-note">{txt}</div>', unsafe_allow_html=True)

def require_derived(names):
    missing = [n for n in names if not _exists(_derived_path(DERIVED_FILES[n]))]
    if missing:
        st.error(
            "Some derived tables are missing. Run `python prep_eda.py` first.\n\nMissing:\n- "
            + "\n- ".join(missing)
        )
        st.stop()

def safe_line(df, x, y, title, height=320):
    fig = px.line(df, x=x, y=y, markers=True, title=title)
    fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=50, b=10), height=height)
    fig.update_traces(line=dict(width=3))
    return fig

# ----------------------------
# Header (no nesting div tricks)
# ----------------------------
st.title("EDA Dashboards")
st.caption(
    "Read-only descriptive analytics for the synthetic SaaS dataset. "
    "In these dashboards, churn is shown using churn_events (event-based) due to known cross-table inconsistencies."
)

# Fixed monthly (per your request)
GRANULARITY = "Monthly"

# ----------------------------
# Sidebar navigation (2 blocks)
# ----------------------------
st.sidebar.title("Menu")

main_modules = [
    "Training Data — EDA Dashboards",
    "Upload & Modeling",
]
main_choice = st.sidebar.radio("Sections", main_modules, index=0)

eda_pages = {
    "Executive Overview": "Executive Overview",
    "Revenue & Subscriptions": "Revenue & Subscriptions",
    "Trial & Conversion": "Trial & Conversion",
    "Product Adoption & Usage Quality": "Product Adoption & Usage Quality",
    "Support & Customer Experience": "Support & Customer Experience",
    "Retention & Churn": "Retention & Churn",
    "About This Data": "About This Data",
}

if main_choice == "Training Data — EDA Dashboards":
    st.sidebar.markdown("**Training Data — EDA Dashboards**")
    page = st.sidebar.radio("Pages", list(eda_pages.keys()), index=0)
else:
    page = "Upload & Modeling"

st.divider()

# ============================
# Page: Executive Overview
# ============================
if page == "Executive Overview":
    require_derived(["KPIs Snapshot", "Signups by Month", "MRR by Month", "Tickets by Month", "Churn Events by Month"])

    kpis = load_derived(DERIVED_FILES["KPIs Snapshot"])
    signups = load_derived(DERIVED_FILES["Signups by Month"])
    mrr = load_derived(DERIVED_FILES["MRR by Month"])
    tickets = load_derived(DERIVED_FILES["Tickets by Month"])
    churn = load_derived(DERIVED_FILES["Churn Events by Month"])

    # KPI row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        rs_card("Accounts", fmt_int(kpis.loc[0, "accounts"]), "Unique customers")
    with c2:
        rs_card("Subscription records", fmt_int(kpis.loc[0, "subscription_records"]), "Lifecycle records")
    with c3:
        rs_card("Tickets", fmt_int(kpis.loc[0, "tickets"]), "Support workload")
    with c4:
        rs_card("Churn events", fmt_int(kpis.loc[0, "churn_events"]), "Event-based churn")
    with c5:
        rs_card("Latest MRR month", str(kpis.loc[0, "latest_mrr_month"]), "From subscriptions")
    with c6:
        rs_card("Latest MRR", fmt_money(kpis.loc[0, "latest_mrr"]), "Sum of MRR (record-month)")

    # Charts
    left, right = st.columns([2, 2], gap="xsmall")

    with left:
        with st.container(border=True):
            st.subheader("Business Activity")
            st.caption("Signups, revenue, and customer lifecycle signals (monthly)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pd.to_datetime(signups["signup_month"]), y=signups["signups"], mode="lines+markers", name="Signups"))
            fig.add_trace(go.Scatter(x=pd.to_datetime(mrr["start_month"]), y=mrr["mrr"], mode="lines+markers", name="MRR"))
            fig.update_layout(
                template="plotly_white",
                margin=dict(l=10, r=10, t=10, b=10),
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(fig, use_container_width=True)

    with right:
        with st.container(border=True):
            st.subheader("Customer Risk Signals")
            st.caption("Support load and churn events")
            fig1 = safe_line(tickets, "submitted_month", "tickets", "Tickets over time", height=300)
            st.plotly_chart(fig1, use_container_width=True)
            fig2 = safe_line(churn, "churn_month", "churn_events", "Churn events over time", height=300)
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        '<div class="rs-note"><b>Interpretation:</b> This overview summarizes growth (signups), monetization (MRR), '
        'customer friction (support tickets), and service termination signals (churn events). '
        'Values reflect a synthetic dataset and are intended for workflow demonstration.</div>',
        unsafe_allow_html=True,
    )

    data_sources_note(
        ["ravenstack_accounts.csv", "ravenstack_subscriptions.csv", "ravenstack_support_tickets.csv", "ravenstack_churn_events.csv"],
        definitions="Churn shown here is based on churn_events (event-based).",
    )

# ============================
# Page: Revenue & Subscriptions
# ============================
elif page == "Revenue & Subscriptions":
    require_derived(["MRR by Plan by Month", "Plan Changes by Month", "Billing Split"])

    mrr_plan = load_derived(DERIVED_FILES["MRR by Plan by Month"])
    changes = load_derived(DERIVED_FILES["Plan Changes by Month"])
    billing = load_derived(DERIVED_FILES["Billing Split"])

    total_mrr = float(mrr_plan["mrr"].sum()) if "mrr" in mrr_plan.columns else 0.0
    upgrades = int(changes["upgrades"].sum()) if "upgrades" in changes.columns else 0
    downgrades = int(changes["downgrades"].sum()) if "downgrades" in changes.columns else 0

    a, b, c = st.columns(3)
    with a:
        rs_card("Total MRR (sum)", fmt_money(total_mrr), "Across subscription records", pill_text="EDA", pill_kind="blue")
    with b:
        rs_card("Upgrade signals", fmt_int(upgrades), "Count of upgrade_flag=True", pill_text="Events", pill_kind="green")
    with c:
        rs_card("Downgrade signals", fmt_int(downgrades), "Count of downgrade_flag=True", pill_text="Events", pill_kind="red")

    col1, col2 = st.columns([2, 1.2], gap="xsmall")

    with col1:
        with st.container(border=True):
            st.subheader("MRR by Plan Tier")
            st.caption("Revenue composition over time (monthly)")
            mrr_plan["start_month"] = pd.to_datetime(mrr_plan["start_month"], errors="coerce")
            dfp = mrr_plan.dropna(subset=["start_month"]).copy()
            fig = px.area(dfp, x="start_month", y="mrr", color="plan_tier", title="")
            fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=10, b=10), height=360, legend=dict(orientation="h", y=1.02, x=0))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        with st.container(border=True):
            st.subheader("Billing Frequency")
            st.caption("Share of monthly vs annual subscriptions")
            fig = px.pie(billing, names="billing_frequency", values="count", title="")
            fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=10, b=10), height=360)
            st.plotly_chart(fig, use_container_width=True)

    with st.container(border=True):
        st.subheader("Plan Change Signals")
        st.caption("Upgrade / downgrade counts over time (monthly)")
        changes["start_month"] = pd.to_datetime(changes["start_month"], errors="coerce")
        fig = px.line(changes, x="start_month", y=["upgrades", "downgrades"], markers=True, title="")
        fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=10, b=10), height=340, legend=dict(orientation="h", y=1.02, x=0))
        fig.update_traces(line=dict(width=3))
        st.plotly_chart(fig, use_container_width=True)

    data_sources_note(
        ["ravenstack_subscriptions.csv"],
        definitions="MRR is aggregated from subscriptions.mrr_amount using the month of start_date (record-based trend).",
    )

# ============================
# Page: Trial & Conversion
# ============================
elif page == "Trial & Conversion":
    if not _exists(_raw_path(RAW_FILES["Accounts"])) or not _exists(_raw_path(RAW_FILES["Subscriptions"])):
        st.error("Missing raw files in data/raw/. Please place ravenstack_accounts.csv and ravenstack_subscriptions.csv in data/raw/.")
        st.stop()

    st.markdown(
        '<div class="rs-note"><b>Purpose:</b> Describe trial exposure, trial→paid conversion, and conversion timing using a simple, transparent rule (EDA only).</div>',
        unsafe_allow_html=True
    )

    accounts = load_raw(RAW_FILES["Accounts"])
    subs = load_raw(RAW_FILES["Subscriptions"])

    accounts["signup_date"] = pd.to_datetime(accounts.get("signup_date"), errors="coerce")
    accounts["signup_month"] = accounts["signup_date"].dt.to_period("M").dt.to_timestamp()

    subs["start_date"] = pd.to_datetime(subs.get("start_date"), errors="coerce")
    subs["mrr_amount"] = pd.to_numeric(subs.get("mrr_amount"), errors="coerce").fillna(0)
    subs["is_trial"] = subs.get("is_trial").astype(bool)

    trial_subs = subs[subs["is_trial"] == True].copy()
    first_trial = trial_subs.groupby("account_id", as_index=False).agg(first_trial_start=("start_date", "min"))

    paid_subs = subs[(subs["is_trial"] == False) & (subs["mrr_amount"] > 0)].copy()
    first_paid = paid_subs.groupby("account_id", as_index=False).agg(first_paid_start=("start_date", "min"))

    conv = accounts.merge(first_trial, on="account_id", how="left").merge(first_paid, on="account_id", how="left")
    conv["had_trial"] = conv["first_trial_start"].notna() | conv.get("is_trial").astype(bool)
    conv["converted_to_paid"] = conv["first_paid_start"].notna()
    conv["days_to_convert"] = (conv["first_paid_start"] - conv["signup_date"]).dt.days
    conv.loc[conv["converted_to_paid"] == False, "days_to_convert"] = pd.NA

    total_accounts = len(conv)
    trial_accounts = int(conv["had_trial"].sum())
    converted_accounts = int(conv["converted_to_paid"].sum())
    conversion_rate = (converted_accounts / trial_accounts) if trial_accounts else 0.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        rs_card("Total accounts", fmt_int(total_accounts), "All customers in accounts.csv")
    with c2:
        rs_card("Trial exposure", fmt_int(trial_accounts), "Had a trial signal", pill_text="EDA", pill_kind="blue")
    with c3:
        rs_card("Converted to paid", fmt_int(converted_accounts), "Paid signal detected", pill_text="EDA", pill_kind="green")
    with c4:
        rs_card("Trial→Paid rate", f"{conversion_rate*100:.1f}%", "Converted / Trial-exposed", pill_text="Rule-based", pill_kind="blue")

    cohort = conv.groupby("signup_month", as_index=False).agg(
        accounts=("account_id", "count"),
        had_trial=("had_trial", "sum"),
        converted=("converted_to_paid", "sum"),
    )
    cohort["conversion_rate_vs_trial"] = cohort.apply(
        lambda r: (r["converted"] / r["had_trial"]) if r["had_trial"] else 0.0,
        axis=1
    )
    cohort = cohort.sort_values("signup_month")

    left, right = st.columns([2, 2], gap="xsmall")
    with left:
        with st.container(border=True):
            st.subheader("Cohort Conversion Rate")
            st.caption("Trial→paid conversion by signup month cohort (monthly)")
            fig = px.line(cohort, x="signup_month", y="conversion_rate_vs_trial", markers=True, title="")
            fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=10, b=10), height=340)
            fig.update_yaxes(tickformat=".0%")
            fig.update_traces(line=dict(width=3))
            st.plotly_chart(fig, use_container_width=True)

    with right:
        with st.container(border=True):
            st.subheader("Trial vs Conversion Counts")
            st.caption("Volume view (trial-exposed vs converted)")
            fig = px.bar(cohort, x="signup_month", y=["had_trial", "converted"], barmode="group", title="")
            fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=10, b=10), height=340, legend=dict(orientation="h", y=1.02, x=0))
            st.plotly_chart(fig, use_container_width=True)

    converted_only = conv[conv["converted_to_paid"] == True].copy()
    with st.container(border=True):
        st.subheader("Conversion Timing")
        st.caption("How long it took converted accounts to become paying customers (days)")
        if len(converted_only) > 0:
            fig = px.histogram(converted_only, x="days_to_convert", nbins=20, title="")
            fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=10, b=10), height=320)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No converted accounts detected under the current EDA logic.")

    trial_only = conv[conv["had_trial"] == True].copy()

    def segment_rate(df, col):
        seg = df.groupby(col, as_index=False).agg(
            trial_accounts=("account_id", "count"),
            converted=("converted_to_paid", "sum")
        )
        seg["conversion_rate"] = seg["converted"] / seg["trial_accounts"]
        return seg.sort_values(["conversion_rate", "trial_accounts"], ascending=[False, False])

    with st.container(border=True):
        st.subheader("Segment Conversion")
        st.caption("Conversion rate among trial-exposed accounts (top segments)")
        s1, s2, s3 = st.columns(3, gap="xsmall")

        with s1:
            seg = segment_rate(trial_only, "referral_source").head(10)
            fig = px.bar(seg, x="referral_source", y="conversion_rate", title="By referral source")
            fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=40, b=10), height=300)
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

        with s2:
            seg = segment_rate(trial_only, "industry").head(10)
            fig = px.bar(seg, x="industry", y="conversion_rate", title="By industry (top 10)")
            fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=40, b=10), height=300)
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

        with s3:
            seg = segment_rate(trial_only, "country").head(10)
            fig = px.bar(seg, x="country", y="conversion_rate", title="By country (top 10)")
            fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=40, b=10), height=300)
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

    data_sources_note(
        ["ravenstack_accounts.csv", "ravenstack_subscriptions.csv"],
        definitions=(
            "Trial exposure: any subscription is_trial=True OR accounts.is_trial=True. "
            "Paid conversion: any subscription is_trial=False AND mrr_amount>0. "
            "Days-to-convert: first_paid_start_date - signup_date."
        )
    )

# ============================
# Page: Product Adoption & Usage Quality
# ============================
elif page == "Product Adoption & Usage Quality":
    require_derived(["Usage by Month", "Top Features"])

    usage_month = load_derived(DERIVED_FILES["Usage by Month"])
    top_features = load_derived(DERIVED_FILES["Top Features"])

    total_usage = float(usage_month["total_usage_count"].sum()) if "total_usage_count" in usage_month.columns else 0.0
    total_errors = float(usage_month["total_errors"].sum()) if "total_errors" in usage_month.columns else 0.0
    beta_records = float(usage_month["beta_records"].sum()) if "beta_records" in usage_month.columns else 0.0
    total_records = float(usage_month["usage_records"].sum()) if "usage_records" in usage_month.columns else 0.0
    beta_share = (beta_records / total_records) if total_records else 0.0

    a, b, c = st.columns(3)
    with a:
        rs_card("Total usage (count)", f"{total_usage:,.0f}", "Sum of usage_count", pill_text="EDA", pill_kind="blue")
    with b:
        rs_card("Total errors", f"{total_errors:,.0f}", "Sum of error_count", pill_text="Quality", pill_kind="red")
    with c:
        rs_card("Beta share", f"{beta_share*100:.1f}%", "Beta records / usage records", pill_text="Beta", pill_kind="blue")

    col1, col2 = st.columns([2, 2], gap="xsmall")

    with col1:
        with st.container(border=True):
            st.subheader("Usage Volume Over Time")
            st.caption("Total usage_count by month")
            fig = safe_line(usage_month, "usage_month", "total_usage_count", "", height=340)
            fig.update_layout(title="")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        with st.container(border=True):
            st.subheader("Errors Over Time")
            st.caption("Total errors by month")
            fig = safe_line(usage_month, "usage_month", "total_errors", "", height=340)
            fig.update_layout(title="")
            st.plotly_chart(fig, use_container_width=True)

    with st.container(border=True):
        st.subheader("Top Features")
        st.caption("Top 15 features by total usage_count (proxy for adoption)")
        fig = px.bar(top_features, x="feature_name", y="total_usage_count", title="")
        fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=10, b=10), height=360)
        st.plotly_chart(fig, use_container_width=True)

    data_sources_note(
        ["ravenstack_feature_usage.csv", "ravenstack_subscriptions.csv"],
        definitions="Feature usage is recorded by subscription_id; account-level views require joining through subscriptions.",
    )

# ============================
# Page: Support & Customer Experience
# ============================
elif page == "Support & Customer Experience":
    require_derived(["Tickets by Month", "Ticket Priority Dist", "Satisfaction Coverage"])

    tickets = load_derived(DERIVED_FILES["Tickets by Month"])
    priority = load_derived(DERIVED_FILES["Ticket Priority Dist"])
    coverage = load_derived(DERIVED_FILES["Satisfaction Coverage"])

    total_tickets = int(coverage.loc[0, "total_tickets"])
    coverage_rate = float(coverage.loc[0, "satisfaction_coverage_rate"])
    escalations_total = int(tickets["escalations"].sum()) if "escalations" in tickets.columns else 0

    a, b, c = st.columns(3)
    with a:
        rs_card("Total tickets", fmt_int(total_tickets), "All support cases", pill_text="EDA", pill_kind="blue")
    with b:
        rs_card("Escalations (sum)", fmt_int(escalations_total), "escalation_flag=True", pill_text="Risk", pill_kind="red")
    with c:
        rs_card("Satisfaction coverage", f"{coverage_rate*100:.1f}%", "Share of tickets with a score", pill_text="CX", pill_kind="green")

    col1, col2 = st.columns([2, 2], gap="xsmall")

    with col1:
        with st.container(border=True):
            st.subheader("Ticket Volume Over Time")
            st.caption("Tickets by month")
            fig = safe_line(tickets, "submitted_month", "tickets", "", height=340)
            fig.update_layout(title="")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        with st.container(border=True):
            st.subheader("Priority Distribution")
            st.caption("How urgent tickets tend to be")
            fig = px.bar(priority, x="priority", y="count", title="")
            fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=10, b=10), height=340)
            st.plotly_chart(fig, use_container_width=True)

    with st.container(border=True):
        st.subheader("Satisfaction Coverage Detail")
        st.caption("Missing scores mean the customer did not submit a rating")
        st.dataframe(coverage, use_container_width=True)

    data_sources_note(["ravenstack_support_tickets.csv"], definitions="Satisfaction metrics are computed from non-null satisfaction_score values.")

# ============================
# Page: Retention & Churn
# ============================
elif page == "Retention & Churn":
    require_derived(["Churn Events by Month", "Churn Reasons", "Reactivation Summary"])

    churn_month = load_derived(DERIVED_FILES["Churn Events by Month"])
    reasons = load_derived(DERIVED_FILES["Churn Reasons"])
    react = load_derived(DERIVED_FILES["Reactivation Summary"])

    total_churn = int(react.loc[0, "total_churn_events"])
    reactivation_rate = float(react.loc[0, "reactivation_rate"])
    refunds_positive = int(churn_month["refunds_positive"].sum()) if "refunds_positive" in churn_month.columns else 0

    a, b, c = st.columns(3)
    with a:
        rs_card("Churn events", fmt_int(total_churn), "Event-based churn", pill_text="EDA", pill_kind="blue")
    with b:
        rs_card("Reactivation rate", f"{reactivation_rate*100:.1f}%", "is_reactivation=True share", pill_text="Lifecycle", pill_kind="green")
    with c:
        rs_card("Refund cases (count)", fmt_int(refunds_positive), "refund_amount_usd > 0", pill_text="Signals", pill_kind="red")

    col1, col2 = st.columns([2, 2], gap="xsmall")

    with col1:
        with st.container(border=True):
            st.subheader("Churn Events Over Time")
            st.caption("Churn events by month")
            fig = safe_line(churn_month, "churn_month", "churn_events", "", height=340)
            fig.update_layout(title="")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        with st.container(border=True):
            st.subheader("Churn Reasons")
            st.caption("Most common reason_code values")
            rr = reasons.sort_values("count", ascending=False).head(10)
            fig = px.bar(rr, x="reason_code", y="count", title="")
            fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=10, b=10), height=340)
            st.plotly_chart(fig, use_container_width=True)

    with st.container(border=True):
        st.subheader("Reactivation Summary")
        st.caption("Basic reactivation statistics")
        st.dataframe(react, use_container_width=True)

    data_sources_note(
        ["ravenstack_churn_events.csv"],
        definitions="Churn is measured using churn_events for EDA consistency (event-based)."
    )

# ============================
# Page: About This Data
# ============================
elif page == "About This Data":
    with st.container(border=True):
        st.subheader("About the Dataset")
        st.caption("Synthetic multi-table SaaS data (no PII)")
        st.markdown(
            """
**What this is:** A synthetic dataset designed to support SQL joins, SaaS analytics, and churn/retention exploration.

**Known limitations (important):**
- Churn indicators are not perfectly consistent across tables (`accounts.churn_flag`, `subscriptions.churn_flag`, and `churn_events`).
- Subscription records behave like event-driven “records/changes” rather than a perfectly continuous lifecycle history.
- Minor synthetic generation artifacts may exist (e.g., a small number of duplicated IDs in usage logs).

**Why this is acceptable for the current demo:**
- The demo goal is to showcase an analytics-friendly frontend and an end-to-end workflow pattern.
- Predictive accuracy and real-world label fidelity are not the primary evaluation criteria at this stage.

**Dashboard churn definition (EDA):**
- Churn is shown using `ravenstack_churn_events.csv` (event-based churn).
            """
        )

    data_sources_note(["All five CSV files"])

# ============================
# Page: Upload & Modeling (Placeholder + Kaggle-style explorer shell)
# ============================
elif page == "Upload & Modeling":
    with st.container(border=True):
        st.subheader("Upload & Modeling")
        st.caption("Placeholder module — will be wired to the modeling pipeline once available")
        st.markdown(
            """
This section is intentionally a **scaffold** so the demo can show the product structure now and plug in the modeling code later.

Planned workflow:
1) Upload CSV/XLSX  
2) Column mapping (ID fields, date fields, numeric/categorical fields)  
3) Confirm schema & data profile (**Kaggle-style explorer**)  
4) Compute outputs (risk table + recommended actions)  
5) Download results
            """
        )

    with st.container(border=True):
        st.subheader("Data Explorer (Prototype)")
        st.caption("Compact preview + Schema & Column Profile")

        raw_choices = [f"RAW • {k}" for k in RAW_FILES.keys()]
        derived_choices = [f"DERIVED • {k}" for k in DERIVED_FILES.keys()]
        file_choice = st.selectbox("Select a file to explore", raw_choices + derived_choices)

        df = None
        file_label = ""

        if file_choice.startswith("RAW"):
            k = file_choice.replace("RAW • ", "")
            fname = RAW_FILES[k]
            file_label = fname
            if _exists(_raw_path(fname)):
                df = load_raw(fname)
            else:
                st.error(f"Missing {fname} in data/raw/.")
        else:
            k = file_choice.replace("DERIVED • ", "")
            fname = DERIVED_FILES[k]
            file_label = fname
            if _exists(_derived_path(fname)):
                df = load_derived(fname)
            else:
                st.error(f"Missing {fname} in data/derived/. Run `python prep_eda.py`.")

        tab1, tab2 = st.tabs(["Compact", "Schema & Column Profile"])

        with tab1:
            st.markdown(f"**File:** `{file_label}`")
            if df is not None:
                c1, c2, c3 = st.columns(3)
                with c1:
                    rs_card("Rows", fmt_int(len(df)), "Record count")
                with c2:
                    rs_card("Columns", fmt_int(df.shape[1]), "Field count")
                with c3:
                    missing_cells = int(df.isna().sum().sum())
                    total_cells = int(df.shape[0] * df.shape[1])
                    miss_rate = (missing_cells / total_cells) if total_cells else 0.0
                    rs_card(
                        "Missing cells",
                        fmt_int(missing_cells),
                        f"{miss_rate*100:.1f}% of all cells",
                        pill_text="Health",
                        pill_kind=("red" if miss_rate > 0.10 else "green"),
                    )
                st.markdown("**Preview (first 50 rows):**")
                st.dataframe(df.head(50), use_container_width=True)

        with tab2:
            if df is not None:
                def infer_type(s: pd.Series) -> str:
                    if pd.api.types.is_bool_dtype(s):
                        return "Boolean"
                    if pd.api.types.is_integer_dtype(s):
                        return "Integer"
                    if pd.api.types.is_float_dtype(s):
                        return "Float"
                    if s.dtype == object:
                        parsed = pd.to_datetime(s, errors="coerce")
                        if parsed.notna().mean() > 0.8:
                            has_time = (parsed.dt.hour.fillna(0) != 0).mean() > 0.05
                            return "DateTime" if has_time else "Date"
                    return "Text"

                profile_rows = []
                n = len(df)
                for col in df.columns:
                    s = df[col]
                    missing = int(s.isna().sum())
                    missing_rate = missing / n if n else 0.0
                    uniq = int(s.nunique(dropna=True))
                    dtype = infer_type(s)

                    most_common_val = ""
                    most_common_rate = np.nan
                    vc = s.value_counts(dropna=True)
                    if len(vc) > 0:
                        most_common_val = str(vc.index[0])
                        denom = (n - missing)
                        most_common_rate = float(vc.iloc[0] / denom) if denom else np.nan

                    profile_rows.append({
                        "column": col,
                        "inferred_type": dtype,
                        "missing": missing,
                        "missing_rate": missing_rate,
                        "unique_values": uniq,
                        "most_common": most_common_val,
                        "most_common_rate": most_common_rate,
                    })

                prof = pd.DataFrame(profile_rows)
                st.dataframe(prof, use_container_width=True)

                st.markdown("---")
                col_sel = st.selectbox("Pick a column", df.columns.tolist())
                s = df[col_sel]
                dtype = infer_type(s)
                missing = int(s.isna().sum())
                uniq = int(s.nunique(dropna=True))

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    rs_card("Inferred type", dtype, "Heuristic inference")
                with c2:
                    rs_card("Missing", fmt_int(missing), f"{(missing/n*100 if n else 0):.1f}%")
                with c3:
                    rs_card("Unique", fmt_int(uniq), f"{(uniq/n*100 if n else 0):.1f}% of rows")
                with c4:
                    id_like = (uniq / n) > 0.95 if n else False
                    rs_card("Heuristic", "ID-like" if id_like else "Not ID-like", "Based on uniqueness")

                with st.container(border=True):
                    st.subheader("Distribution Preview")
                    st.caption("Chart type chosen by inferred column type")

                    if dtype in ("Integer", "Float"):
                        vals = pd.to_numeric(s, errors="coerce").dropna()
                        if len(vals) > 0:
                            fig = px.histogram(vals, nbins=30, title="")
                            fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=10, b=10), height=320)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("No numeric values to plot.")

                    elif dtype in ("Date", "DateTime"):
                        dt = pd.to_datetime(s, errors="coerce").dropna()
                        if len(dt) > 0:
                            by_m = dt.dt.to_period("M").dt.to_timestamp().value_counts().sort_index()
                            tmp = pd.DataFrame({"period": by_m.index, "count": by_m.values})
                            fig = px.bar(tmp, x="period", y="count", title="")
                            fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=10, b=10), height=320)
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption(f"Range: {dt.min().date()} → {dt.max().date()}")
                        else:
                            st.warning("No parsable dates to plot.")

                    elif dtype == "Boolean":
                        vc = s.astype("boolean").value_counts(dropna=True)
                        tmp = pd.DataFrame({"value": vc.index.astype(str), "count": vc.values})
                        fig = px.pie(tmp, names="value", values="count", title="")
                        fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=10, b=10), height=320)
                        st.plotly_chart(fig, use_container_width=True)

                    else:
                        vc = s.value_counts(dropna=True).head(15)
                        if len(vc) > 0:
                            tmp = pd.DataFrame({"value": vc.index.astype(str), "count": vc.values})
                            fig = px.bar(tmp, x="value", y="count", title="")
                            fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=10, b=10), height=320)
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption("Top 15 most common values (categorical-like preview).")
                        else:
                            st.warning("No values to summarize.")

    data_sources_note(
        ["(Prototype) raw + derived CSVs"],
        definitions="This explorer is a UI scaffold inspired by Kaggle-style file exploration: Compact preview + column profiling."
    )
