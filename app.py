"""EduPulse AI Streamlit dashboard."""
from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from edupulse_ml import (
    MODEL_PATH, load_bundle, make_predictions, risk_label,
    suggest_excluded_columns, train_regression_model,
)
from generate_data import generate_student_data

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "student_performance_demo.csv"


@st.cache_data
def get_demo_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    frame = generate_student_data()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DATA_PATH, index=False)
    return frame


@st.cache_resource
def get_demo_bundle():
    if MODEL_PATH.exists():
        return load_bundle(MODEL_PATH)
    frame = get_demo_data()
    bundle = train_regression_model(
        frame, "final_score",
        suggest_excluded_columns(frame.columns.tolist(), "final_score"),
    )
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)
    return bundle


def read_table(upload):
    if upload.name.lower().endswith(".csv"):
        return pd.read_csv(upload)
    return pd.read_excel(upload)


def number(value, digits=1):
    return "—" if pd.isna(value) else f"{float(value):,.{digits}f}"


def model_bytes(bundle):
    buffer = BytesIO()
    joblib.dump(bundle, buffer, compress=3)
    return buffer.getvalue()


st.set_page_config(page_title="EduPulse AI", page_icon="🎓", layout="wide")
st.markdown("""
<style>
.block-container{max-width:1500px;padding-top:1.5rem}
.hero{padding:24px;border-radius:16px;background:linear-gradient(110deg,#163b70,#147d9a);color:white;margin-bottom:18px}
.hero h1{margin:0}.hero p{margin:6px 0 0;opacity:.9}
[data-testid="stMetric"]{padding:12px;border:1px solid rgba(128,128,128,.2);border-radius:12px}
</style>
<div class="hero"><h1>🎓 EduPulse AI</h1><p>Student performance analytics, predictive insights and early academic support signals.</p></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("EduPulse AI")
    upload = st.file_uploader("Upload student data", type=["csv", "xlsx", "xls"])
    page = st.radio("Workspace", [
        "Executive Overview", "Data Explorer", "Model Training & Evaluation",
        "Individual Prediction", "Batch Predictions", "Model Card",
    ])

try:
    data = read_table(upload) if upload else get_demo_data()
    demo_bundle = get_demo_bundle()
except Exception as exc:
    st.error(f"Could not initialize EduPulse AI: {exc}")
    st.stop()

if data.empty:
    st.error("The selected dataset is empty.")
    st.stop()

if "active_bundle" not in st.session_state:
    st.session_state.active_bundle = None
bundle = st.session_state.active_bundle or demo_bundle
model_label = "Custom trained model" if st.session_state.active_bundle else "Demo Random Forest"

st.caption(f"{len(data):,} records · {model_label} · {'Uploaded dataset' if upload else 'Synthetic demo dataset'}")

score_col = next((c for c in ["final_score", "score", "marks", "percentage"] if c in data.columns and pd.api.types.is_numeric_dtype(data[c])), None)
attendance_col = next((c for c in data.columns if c.lower() in {"attendance_rate", "attendance", "attendance_percentage"}), None)

if page == "Executive Overview":
    st.subheader("Academic performance at a glance")
    if score_col:
        scores = pd.to_numeric(data[score_col], errors="coerce")
        a, b, c, d = st.columns(4)
        a.metric("Average score", f"{number(scores.mean())} / 100")
        b.metric("Pass rate", f"{number((scores >= 40).mean()*100)}%")
        c.metric("Top score", number(scores.max()))
        d.metric("Below 40", f"{int((scores < 40).sum()):,}")
        left, right = st.columns(2)
        with left:
            st.markdown("#### Score distribution")
            st.plotly_chart(px.histogram(data, x=score_col, nbins=24), use_container_width=True)
        with right:
            st.markdown("#### Department performance")
            if "department" in data.columns:
                summary = data.groupby("department", dropna=False)[score_col].mean().reset_index()
                st.plotly_chart(px.bar(summary, x="department", y=score_col, color=score_col), use_container_width=True)
            else:
                st.info("Add a department column to see this comparison.")
    else:
        st.info("No standard numeric score column detected.")
        st.metric("Student records", f"{len(data):,}")
    if attendance_col:
        st.metric("Average attendance", f"{number(pd.to_numeric(data[attendance_col], errors='coerce').mean())}%")
    st.markdown("#### Student data preview")
    st.dataframe(data.head(15), use_container_width=True, hide_index=True)

elif page == "Data Explorer":
    st.subheader("Explore and validate student data")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(data):,}")
    c2.metric("Columns", f"{data.shape[1]:,}")
    c3.metric("Missing cells", f"{int(data.isna().sum().sum()):,}")
    tabs = st.tabs(["Preview", "Data quality", "Relationships"])
    with tabs[0]:
        query = st.text_input("Search across rows")
        view = data
        if query.strip():
            mask = data.astype(str).apply(lambda col: col.str.contains(query, case=False, na=False)).any(axis=1)
            view = data.loc[mask]
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.download_button("Download dataset CSV", data.to_csv(index=False).encode(), "edupulse_dataset.csv", "text/csv")
    with tabs[1]:
        quality = pd.DataFrame({"column": data.columns, "dtype": data.dtypes.astype(str).values,
                                "missing": data.isna().sum().values,
                                "unique": data.nunique(dropna=True).values})
        st.dataframe(quality, use_container_width=True, hide_index=True)
    with tabs[2]:
        numeric = data.select_dtypes(include="number")
        if numeric.shape[1] >= 2:
            st.plotly_chart(px.imshow(numeric.corr(), text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r"), use_container_width=True)
        else:
            st.info("At least two numeric columns are required for correlation analysis.")

elif page == "Model Training & Evaluation":
    st.subheader("Train and evaluate a prediction model")
    targets = data.select_dtypes(include="number").columns.tolist()
    if not targets:
        st.warning("Upload data containing a numeric target column.")
    else:
        default = "final_score" if "final_score" in targets else targets[-1]
        target = st.selectbox("Prediction target", targets, index=targets.index(default))
        excluded_default = suggest_excluded_columns(data.columns.tolist(), target)
        excluded = st.multiselect("Columns to exclude", [c for c in data.columns if c != target], default=excluded_default,
                                  help="Exclude IDs and information that leaks the outcome.")
        if st.button("Train / Retrain model", type="primary"):
            try:
                with st.spinner("Training Random Forest and evaluating held-out records..."):
                    st.session_state.active_bundle = train_regression_model(data, target, excluded)
                st.success("Training completed. The custom model is active for this session.")
                st.rerun()
            except Exception as exc:
                st.error(f"Training failed: {exc}")
    st.divider()
    st.markdown("#### Active model evaluation")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", number(bundle["metrics"]["mae"], 2))
    m2.metric("RMSE", number(bundle["metrics"]["rmse"], 2))
    m3.metric("R²", number(bundle["metrics"]["r2"], 3))
    m4.metric("Training records", f"{bundle['sample_count']:,}")
    results = bundle["test_results"]
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.scatter(results, x="actual", y="predicted", title="Actual vs predicted"), use_container_width=True)
    with right:
        st.plotly_chart(px.histogram(results, x="residual", title="Residual distribution"), use_container_width=True)
    importance = pd.DataFrame(bundle["feature_importance"])
    if not importance.empty:
        st.plotly_chart(px.bar(importance.sort_values("importance"), x="importance", y="feature", orientation="h", title="Feature importance"), use_container_width=True)
    st.download_button("Download active model (.joblib)", model_bytes(bundle), "edupulse_model.joblib", "application/octet-stream")
    if st.session_state.active_bundle and st.button("Reset to demo model"):
        st.session_state.active_bundle = None
        st.rerun()

elif page == "Individual Prediction":
    st.subheader("Estimate an individual student's outcome")
    st.info("Enter information available before the outcome is known. Predictions are estimates, not decisions.")
    values = {}
    with st.form("student_prediction"):
        columns = st.columns(2)
        for i, feature in enumerate(bundle["feature_columns"]):
            meta = bundle["feature_metadata"][feature]
            with columns[i % 2]:
                label = feature.replace("_", " ").title()
                if meta["kind"] == "numeric":
                    low, high = float(meta["minimum"]), float(meta["maximum"])
                    pad = max((high-low)*.05, .5)
                    values[feature] = st.number_input(label, min_value=low-pad, max_value=high+pad,
                                                       value=float(meta["median"]), step=1.0 if meta.get("integer") else .1)
                else:
                    choices = [str(x) for x in meta["choices"]]
                    values[feature] = st.selectbox(label, choices)
        submitted = st.form_submit_button("Predict outcome", type="primary")
    if submitted:
        row = pd.DataFrame([values], columns=bundle["feature_columns"])
        prediction = float(bundle["model"].predict(row)[0])
        risk = risk_label(prediction, bundle["risk_thresholds"])
        c1, c2 = st.columns(2)
        c1.metric(f"Predicted {bundle['target_column']}", number(prediction, 2))
        c2.metric("Estimated risk", risk)
        st.info("Review this estimate with the student and other relevant evidence.")

elif page == "Batch Predictions":
    st.subheader("Batch student predictions")
    st.caption("Upload a CSV or Excel file containing all required feature columns.")
    st.code(", ".join(bundle["feature_columns"]))
    batch = st.file_uploader("Choose batch file", type=["csv", "xlsx", "xls"], key="batch")
    if batch:
        try:
            batch_data = read_table(batch)
            missing = [c for c in bundle["feature_columns"] if c not in batch_data.columns]
            if missing:
                st.error("Missing required columns: " + ", ".join(missing))
            elif st.button("Generate predictions", type="primary"):
                output = make_predictions(bundle, batch_data)
                st.dataframe(output, use_container_width=True, hide_index=True)
                st.download_button("Download predictions CSV", output.to_csv(index=False).encode(),
                                   "edupulse_predictions.csv", "text/csv")
                counts = output["predicted_risk"].value_counts().rename_axis("risk").reset_index(name="students")
                st.plotly_chart(px.pie(counts, names="risk", values="students", hole=.45), use_container_width=True)
        except Exception as exc:
            st.error(f"Could not process file: {exc}")

elif page == "Model Card":
    st.subheader("Model card and responsible-use notes")
    st.write(f"**Model:** {bundle['model_name']} · **Version:** {bundle['model_version']}")
    st.write(f"**Target:** {bundle['target_column']} · **Features:** {len(bundle['feature_columns'])}")
    st.json(bundle["metrics"])
    st.markdown("#### Intended use")
    st.write("An additional signal for academic support planning, cohort monitoring and student progress conversations.")
    st.warning("The bundled model uses synthetic demonstration data. Its metrics do not represent real-world accuracy. Validate on representative, consented institutional data and audit subgroup errors, privacy and drift. Never use a prediction as the sole basis for high-impact academic decisions.")
    st.dataframe(pd.DataFrame({"feature": bundle["feature_columns"]}), use_container_width=True, hide_index=True)

st.sidebar.divider()
st.sidebar.caption("EduPulse AI · Academic analytics prototype")
st.sidebar.caption("Predictions support human review; they do not replace educators.")
