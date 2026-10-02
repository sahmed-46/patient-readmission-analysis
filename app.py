from pathlib import Path

import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import streamlit as st


DATA_PATH = Path(__file__).with_name("hospital_readmissions.csv")
AGE_ORDER = ["[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
FILTERS = {
    "Age band": "age",
    "Medical specialty": "medical_specialty",
    "Diabetes medication": "diabetes_med",
    "Medication change": "change",
    "A1C test": "A1Ctest",
    "Glucose test": "glucose_test",
    "Readmission outcome": "readmitted",
}

st.set_page_config(
    page_title="Hospital Readmissions | Outcomes Dashboard",
    page_icon="H",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root {
        --ink: #172b2a;
        --muted: #657875;
        --paper: #f4f7f4;
        --line: #dbe4df;
        --green: #087e72;
        --coral: #d8674e;
    }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
    .stApp { background: var(--paper); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: #e8efeb; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] > div { padding-top: 1.4rem; }
    .eyebrow { font: 500 11px 'DM Mono', monospace; letter-spacing: 0; color: var(--green); text-transform: uppercase; }
    .page-title { font: 800 38px 'Manrope', sans-serif; line-height: 1.1; margin: 7px 0 8px; letter-spacing: 0; }
    .subtitle { color: var(--muted); font-size: 14px; margin-bottom: 24px; }
    .section-label { font: 500 11px 'DM Mono', monospace; color: var(--muted); text-transform: uppercase; letter-spacing: 0; margin: 22px 0 8px; }
    [data-testid="stSidebar"] { width: 290px !important; min-width: 290px !important; max-width: 290px !important; }
    .stMetric { background: #fff; border: 1px solid var(--line); border-radius: 5px; padding: 16px 18px; min-height: 106px; }
    [data-testid="stMetricLabel"] { font-size: 12px; color: var(--muted); }
    [data-testid="stMetricValue"] { font: 700 27px 'Manrope', sans-serif; }
    [data-testid="stMetricDelta"] { font-size: 11px; }
    div[data-testid="stPlotlyChart"] { background: #fff; border: 1px solid var(--line); border-radius: 5px; padding: 8px 10px 0; }
    button, [role="button"], [data-baseweb="select"], input[type="range"],
    .js-plotly-plot .point, .js-plotly-plot .slice, .js-plotly-plot .slice path.surface,
    .js-plotly-plot .modebar-btn, .js-plotly-plot .legendtoggle { cursor: pointer !important; }
    .footnote { color: var(--muted); font: 11px 'DM Mono', monospace; margin-top: 4px; }
    .stSelectbox label, .stSlider label { font-size: 12px; color: var(--ink); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data
def evaluate_logistic_model(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    encoded = pd.get_dummies(frame, prefix_sep="_", drop_first=True).astype(int)
    predictors = encoded.drop(columns=["readmitted_yes"])
    outcome = encoded["readmitted_yes"]
    train_x, valid_x, train_y, valid_y = train_test_split(
        predictors,
        outcome,
        test_size=0.4,
        random_state=1,
    )
    model = LogisticRegression(solver="liblinear", C=1e42)
    model.fit(train_x, train_y)
    predicted = model.predict(valid_x)
    probabilities = model.predict_proba(valid_x)

    predictions = pd.DataFrame(
        {
            "Record": valid_x.index,
            "Actual": valid_y.map({0: "Not readmitted", 1: "Readmitted"}).to_numpy(),
            "Prediction": pd.Series(predicted, index=valid_x.index).map(
                {0: "Not readmitted", 1: "Readmitted"}
            ).to_numpy(),
            "P(not readmitted)": probabilities[:, 0],
            "P(readmitted)": probabilities[:, 1],
        }
    )
    coefficients = pd.DataFrame(
        {"Feature": predictors.columns, "Coefficient": model.coef_[0]}
    ).sort_values("Coefficient", key=lambda values: values.abs(), ascending=False)
    return predictions, coefficients


if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH.name}. Keep it beside app.py.")
    st.stop()

data = load_data()

pending_chart_filter = st.session_state.pop("pending_chart_filter", None)
if pending_chart_filter:
    st.session_state[f"filter_{pending_chart_filter['column']}"] = pending_chart_filter["value"]

st.sidebar.markdown("<div class='eyebrow'>Cohort controls</div>", unsafe_allow_html=True)
st.sidebar.markdown("### Refine the population")
st.sidebar.caption("Filters update metrics and charts in both views.")

selected_values = {}
for label, column in FILTERS.items():
    values = sorted(data[column].dropna().astype(str).unique().tolist())
    if column == "age":
        values = [value for value in AGE_ORDER if value in values]
    selected_values[column] = st.sidebar.selectbox(
        label,
        ["All"] + values,
        index=0,
        key=f"filter_{column}",
    )

duration_min = int(data["time_in_hospital"].min())
duration_max = int(data["time_in_hospital"].max())
stay_range = st.sidebar.slider(
    "Length of stay (days)",
    min_value=duration_min,
    max_value=duration_max,
    value=(duration_min, duration_max),
)
st.sidebar.markdown("<div class='footnote'>Source: hospital_readmissions.csv</div>", unsafe_allow_html=True)


def apply_cohort_filters(frame: pd.DataFrame) -> pd.DataFrame:
    matches = pd.Series(True, index=frame.index)
    for column, value in selected_values.items():
        if value != "All":
            matches &= frame[column].astype(str) == value
    matches &= frame["time_in_hospital"].between(*stay_range)
    return frame.loc[matches]


def apply_chart_filter(chart_state, column: str, point_field: str, value_map=None) -> None:
    points = chart_state.selection.points
    if not points:
        return

    value = points[-1].get(point_field)
    if value_map:
        value = value_map.get(value)
    state_key = f"filter_{column}"
    if value is not None and value != st.session_state.get(state_key, "All"):
        st.session_state["pending_chart_filter"] = {"column": column, "value": value}
        st.rerun()


filtered = apply_cohort_filters(data)

if "active_view" not in st.session_state:
    st.session_state.active_view = "cohort"

cohort_navigation, model_navigation, view_label = st.columns([1, 1, 2])
with cohort_navigation:
    if st.button(
        "Cohort outcomes",
        use_container_width=True,
        type="primary" if st.session_state.active_view == "cohort" else "secondary",
    ):
        if st.session_state.active_view != "cohort":
            st.session_state.active_view = "cohort"
            st.rerun()
with model_navigation:
    if st.button(
        "Model results",
        use_container_width=True,
        type="primary" if st.session_state.active_view == "model" else "secondary",
    ):
        if st.session_state.active_view != "model":
            st.session_state.active_view = "model"
            st.rerun()
with view_label:
    st.markdown(
        f"<div class='footnote'>CURRENT VIEW / {'COHORT OUTCOMES' if st.session_state.active_view == 'cohort' else 'MODEL RESULTS'}</div>",
        unsafe_allow_html=True,
    )

if st.session_state.active_view == "model":
    predictions, coefficients = evaluate_logistic_model(data)
    validation_cohort = apply_cohort_filters(data.loc[predictions["Record"]])
    predictions = predictions[predictions["Record"].isin(validation_cohort.index)]

    if predictions.empty:
        st.markdown("<div class='eyebrow'>Model evaluation / validation partition</div>", unsafe_allow_html=True)
        st.markdown("<div class='page-title'>Model results.</div>", unsafe_allow_html=True)
        st.info("No validation records match these filters. Broaden the cohort to view model results.")
        st.stop()

    actual_values = predictions["Actual"].map({"Not readmitted": 0, "Readmitted": 1})
    predicted_values = predictions["Prediction"].map({"Not readmitted": 0, "Readmitted": 1})
    metrics = {
        "Accuracy": accuracy_score(actual_values, predicted_values),
        "Precision": precision_score(actual_values, predicted_values, zero_division=0),
        "Recall": recall_score(actual_values, predicted_values, zero_division=0),
        "F1 score": f1_score(actual_values, predicted_values, zero_division=0),
    }
    matrix = confusion_matrix(actual_values, predicted_values, labels=[0, 1])

    st.markdown("<div class='eyebrow'>Model evaluation / filtered validation cohort</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-title'>Model results.</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='subtitle'>Logistic regression evaluated on <b>{len(predictions):,}</b> matching records from the notebook's 40% holdout split.</div>",
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(4)
    for column, (label, value) in zip(metric_columns, metrics.items()):
        column.metric(label, f"{value:.1%}")
    st.markdown(
        f"<div class='footnote'>{len(data):,} records / 60% train, 40% validation / random_state=1 / liblinear solver</div>",
        unsafe_allow_html=True,
    )

    matrix_column, coefficient_column = st.columns([1, 1.35])
    with matrix_column:
        matrix_frame = pd.DataFrame(
            matrix,
            index=["Actual: not readmitted", "Actual: readmitted"],
            columns=["Predicted: not readmitted", "Predicted: readmitted"],
        )
        matrix_chart = px.imshow(
            matrix_frame,
            text_auto=True,
            color_continuous_scale=[[0, "#e7efeb"], [1, "#087e72"]],
            labels={"x": "Model prediction", "y": "Observed outcome", "color": "Patients"},
            aspect="auto",
        )
        matrix_chart.update_layout(
            title="Validation confusion matrix",
            clickmode="event+select",
            height=340,
            margin=dict(l=12, r=12, t=48, b=12),
            paper_bgcolor="white",
            font=dict(family="DM Sans, sans-serif", color="#172b2a"),
            coloraxis_showscale=False,
        )
        st.plotly_chart(
            matrix_chart,
            use_container_width=True,
            key="validation_confusion_matrix",
            on_select="rerun",
            selection_mode="points",
        )

    with coefficient_column:
        coefficient_chart = px.bar(
            coefficients.head(12).sort_values("Coefficient"),
            x="Coefficient",
            y="Feature",
            orientation="h",
            color="Coefficient",
            color_continuous_scale=["#d8674e", "#e8efeb", "#087e72"],
            labels={"Coefficient": "Log-odds coefficient", "Feature": "Model feature"},
        )
        coefficient_chart.update_layout(
            title="Largest model coefficients",
            clickmode="event+select",
            height=340,
            margin=dict(l=12, r=12, t=48, b=12),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(family="DM Sans, sans-serif", color="#172b2a"),
            coloraxis_showscale=False,
            xaxis=dict(gridcolor="#e7eeea", zerolinecolor="#657875"),
        )
        st.plotly_chart(
            coefficient_chart,
            use_container_width=True,
            key="model_coefficients_chart",
            on_select="rerun",
            selection_mode="points",
        )

    st.markdown("<div class='section-label'>Validation predictions</div>", unsafe_allow_html=True)
    st.dataframe(
        predictions.head(20).style.format(
            {"P(not readmitted)": "{:.1%}", "P(readmitted)": "{:.1%}"}
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown(
        "<div class='footnote'>Predictions are from the fixed holdout model; filters select a validation subgroup. Model coefficients are fitted on the full training partition and do not change with filters. Associations do not establish causation.</div>",
        unsafe_allow_html=True,
    )
    st.stop()

if st.session_state.active_view == "cohort":
    st.markdown("<div class='eyebrow'>Clinical outcomes / hospital readmissions</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-title'>Readmission, in focus.</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='subtitle'>A filtered view of <b>{len(filtered):,}</b> patient records from a {len(data):,}-record hospital dataset.</div>",
        unsafe_allow_html=True,
    )

    if filtered.empty:
        st.warning("No records match this combination of filters. Broaden the cohort to see results.")
        st.stop()

readmitted = filtered["readmitted"].eq("yes")
readmission_rate = readmitted.mean() * 100
readmitted_count = int(readmitted.sum())
baseline_rate = data["readmitted"].eq("yes").mean() * 100

metric_columns = st.columns(4)
metric_columns[0].metric("Readmission rate", f"{readmission_rate:.1f}%", f"{readmission_rate - baseline_rate:+.1f} pp vs all records")
metric_columns[1].metric("Patients in cohort", f"{len(filtered):,}")
metric_columns[2].metric("Readmitted", f"{readmitted_count:,}", f"{len(filtered) - readmitted_count:,} not readmitted")
metric_columns[3].metric("Median stay", f"{filtered['time_in_hospital'].median():.0f} days")

st.markdown("<div class='section-label'>Readmission patterns</div>", unsafe_allow_html=True)
chart_left, chart_right = st.columns([1.45, 1])

with chart_left:
    age_summary = (
        filtered.assign(_readmitted=readmitted)
        .groupby("age", observed=True)["_readmitted"]
        .agg(rate="mean", patients="size")
        .reset_index()
    )
    age_summary["rate"] *= 100
    age_summary["age"] = pd.Categorical(age_summary["age"], categories=AGE_ORDER, ordered=True)
    age_summary = age_summary.sort_values("age")
    visible_age_order = age_summary["age"].astype(str).tolist()
    age_chart = px.line(
        age_summary,
        x="age",
        y="rate",
        markers=True,
        custom_data=["patients"],
        labels={"age": "Age band", "rate": "Readmission rate", "patients": "Patients"},
    )
    age_chart.update_traces(line_color="#087e72", line_width=3, marker_size=8, hovertemplate="%{x}<br>Rate: %{y:.1f}%<br>Patients: %{customdata[0]}<extra></extra>")
    age_chart.update_layout(
        title="Readmission by age",
        clickmode="event+select",
        height=330,
        margin=dict(l=12, r=16, t=48, b=16),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="DM Sans, sans-serif", color="#172b2a"),
        yaxis=dict(ticksuffix="%", gridcolor="#e7eeea", rangemode="tozero"),
        xaxis=dict(title=None, categoryorder="array", categoryarray=visible_age_order),
        showlegend=False,
    )
    age_selection = st.plotly_chart(
        age_chart,
        use_container_width=True,
        key="readmission_by_age_chart",
        on_select="rerun",
        selection_mode="points",
    )
    apply_chart_filter(age_selection, "age", "x")

with chart_right:
    outcome_summary = filtered["readmitted"].value_counts().rename_axis("outcome").reset_index(name="patients")
    outcome_summary["label"] = outcome_summary["outcome"].map({"yes": "Readmitted", "no": "Not readmitted"})
    outcome_summary["share"] = outcome_summary["patients"] / outcome_summary["patients"].sum()
    outcome_chart = px.scatter(
        outcome_summary,
        x="label",
        y="patients",
        size="patients",
        size_max=34,
        color="label",
        custom_data=["outcome", "share"],
        color_discrete_map={"Readmitted": "#d8674e", "Not readmitted": "#b9d6cb"},
        labels={"label": "Outcome", "patients": "Patients"},
    )
    outcome_chart.update_traces(
        marker=dict(line=dict(color="white", width=2)),
        hovertemplate="%{x}<br>%{y:,} patients (%{customdata[1]:.1%})<extra></extra>",
    )
    outcome_chart.update_layout(
        title="Outcome counts",
        clickmode="event+select",
        height=330,
        margin=dict(l=12, r=12, t=48, b=12),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="DM Sans, sans-serif", color="#172b2a"),
        showlegend=False,
        xaxis=dict(title=None),
        yaxis=dict(title=None, gridcolor="#e7eeea", rangemode="tozero"),
    )
    outcome_selection = st.plotly_chart(
        outcome_chart,
        use_container_width=True,
        key="cohort_outcome_mix_chart",
        on_select="rerun",
        selection_mode="points",
    )
    apply_chart_filter(
        outcome_selection,
        "readmitted",
        "x",
        {"Readmitted": "yes", "Not readmitted": "no"},
    )

st.markdown("<div class='section-label'>Care signals</div>", unsafe_allow_html=True)
signal_left, signal_right = st.columns(2)

with signal_left:
    medication_summary = (
        filtered.assign(_readmitted=readmitted)
        .groupby("diabetes_med", observed=True)["_readmitted"]
        .agg(rate="mean", patients="size")
        .reset_index()
    )
    medication_summary["rate"] *= 100
    medication_chart = px.scatter(
        medication_summary,
        x="diabetes_med",
        y="rate",
        size="patients",
        size_max=30,
        color="diabetes_med",
        color_discrete_map={"yes": "#087e72", "no": "#9ab9ad"},
        custom_data=["patients"],
        labels={"diabetes_med": "Diabetes medication", "rate": "Readmission rate", "patients": "Patients"},
    )
    medication_chart.update_traces(
        marker=dict(line=dict(color="white", width=2)),
        hovertemplate="Medication: %{x}<br>Rate: %{y:.1f}%<br>Patients: %{customdata[0]}<extra></extra>",
    )
    medication_chart.update_layout(
        title="Readmission by medication",
        clickmode="event+select",
        height=310,
        margin=dict(l=12, r=12, t=48, b=12),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="DM Sans, sans-serif", color="#172b2a"),
        showlegend=False,
        xaxis=dict(title=None),
        yaxis=dict(title=None, ticksuffix="%", gridcolor="#e7eeea", rangemode="tozero"),
    )
    medication_selection = st.plotly_chart(
        medication_chart,
        use_container_width=True,
        key="readmission_by_medication_chart",
        on_select="rerun",
        selection_mode="points",
    )
    apply_chart_filter(medication_selection, "diabetes_med", "x")

with signal_right:
    specialty_summary = (
        filtered.assign(_readmitted=readmitted)
        .groupby("medical_specialty", observed=True)["_readmitted"]
        .agg(rate="mean", patients="size")
        .reset_index()
        .sort_values("rate", ascending=True)
    )
    specialty_chart = px.scatter(
        specialty_summary,
        x="rate",
        y="medical_specialty",
        size="patients",
        size_max=24,
        custom_data=["patients"],
        color_discrete_sequence=["#d8674e"],
        labels={"medical_specialty": "Medical specialty", "rate": "Readmission rate", "patients": "Patients"},
    )
    specialty_chart.update_traces(
        marker=dict(line=dict(color="white", width=2)),
        hovertemplate="%{y}<br>Rate: %{x:.1%}<br>Patients: %{customdata[0]}<extra></extra>",
    )
    specialty_chart.update_layout(
        title="Readmission by specialty",
        clickmode="event+select",
        height=310,
        margin=dict(l=12, r=18, t=48, b=12),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="DM Sans, sans-serif", color="#172b2a"),
        showlegend=False,
        xaxis=dict(title=None, tickformat=".0%", gridcolor="#e7eeea", rangemode="tozero"),
        yaxis=dict(title=None),
    )
    specialty_selection = st.plotly_chart(
        specialty_chart,
        use_container_width=True,
        key="readmission_by_specialty_chart",
        on_select="rerun",
        selection_mode="points",
    )
    apply_chart_filter(specialty_selection, "medical_specialty", "y")

st.markdown(
    "<div class='footnote'>Descriptive cohort statistics only. Observed associations do not establish causation or predict individual patient outcomes.</div>",
    unsafe_allow_html=True,
)