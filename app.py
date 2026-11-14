"""Streamlit interface: upload CSV or Excel and interact with the Qwen3 agent."""
from pathlib import Path
import io
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from agent import DataQualityAgent

st.set_page_config(page_title="AI Data Quality Detective", page_icon="🔎", layout="wide")
st.markdown("""
<style>
.stApp {background: #100D1C; color: #F4F0FF;}
[data-testid="stSidebar"] {background: #19142A;}
h1,h2,h3 {color: #C9B6FF !important;}
div.stButton > button {background:#7451D9;color:white;border:0;border-radius:9px;}
div[data-testid="stMetric"] {background:#211A35;padding:14px;border-radius:12px;}
</style>
""", unsafe_allow_html=True)

st.title("🔎 AI-Powered Data Quality Detective")
st.caption("An Ollama + Qwen3 tool-using agent for investigating CSV and Excel datasets.")

with st.sidebar:
    st.header("Agent configuration")
    model = st.text_input("Ollama model", value="qwen3:8b")
    ollama_url = st.text_input("Ollama URL", value="http://localhost:11434")
    st.markdown("---")
    st.markdown("**Agent capabilities**")
    st.write("• Qwen3 planning and explanation")
    st.write("• Four Python analysis tools")
    st.write("• SQLite episodic memory")
    st.write("• Feedback-informed priorities")
    st.write("• Non-destructive cleaning preview")

uploaded = st.file_uploader("Upload your dataset", type=["csv", "xlsx", "xls"])
df = None
if uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            excel = pd.ExcelFile(uploaded)
            sheet = st.selectbox("Choose Excel sheet", excel.sheet_names)
            df = pd.read_excel(uploaded, sheet_name=sheet)
        st.success(f"Loaded {uploaded.name} successfully.")
        st.subheader("Dataset preview")
        st.dataframe(df.head(10), use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", len(df.columns))
        c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    except Exception as e:
        st.error(f"Could not read this file: {e}")

if df is not None:
    goal = st.text_area("What should the agent investigate?", value="Investigate this dataset, identify data quality issues and numeric outlier candidates, and recommend safe next steps.")
    if st.button("🚀 Run intelligent investigation", type="primary"):
        with st.spinner("Qwen3 is planning and the agent is executing tools..."):
            agent = DataQualityAgent(model=model, ollama_url=ollama_url)
            result = agent.investigate(df, goal)
            st.session_state["latest_result"] = result
            st.session_state["agent_model"] = model
            st.session_state["agent_url"] = ollama_url
        st.success("Investigation completed. Review the evidence below.")

    result = st.session_state.get("latest_result")
    if result:
        st.subheader("Agent explanation")
        st.write(result["summary"])
        st.subheader("Execution trace")
        st.dataframe(pd.DataFrame(result["trace"]), use_container_width=True)
        st.caption(f"Execution time: {result['latency_seconds']} seconds | Model: {st.session_state.get('agent_model', model)}")
        observations = result.get("observations", {})
        if "profile_dataset" in observations and "error" not in observations["profile_dataset"]:
            p = observations["profile_dataset"]
            a, b = st.columns(2)
            a.metric("Rows analysed", p["rows"])
            b.metric("Columns analysed", p["columns"])
        if "detect_quality_issues" in observations and "error" not in observations["detect_quality_issues"]:
            q = observations["detect_quality_issues"]
            st.subheader("Data quality findings")
            q1, q2, q3 = st.columns(3)
            q1.metric("Missing cells", q["total_missing_cells"])
            q2.metric("Duplicate rows", q["duplicate_rows"])
            q3.metric("Constant columns", len(q["constant_columns"]))
            if q["missing_values"]:
                st.dataframe(pd.DataFrame.from_dict(q["missing_values"], orient="index").rename_axis("Column"), use_container_width=True)
        if "detect_anomalies" in observations and "error" not in observations["detect_anomalies"]:
            anomaly = observations["detect_anomalies"]["columns"]
            st.subheader("Potential numeric outliers")
            if anomaly:
                chart_df = pd.DataFrame({"Column": list(anomaly.keys()),
                                         "Candidates": [v["candidate_count"] for v in anomaly.values()]})
                st.bar_chart(chart_df.set_index("Column"))
                st.dataframe(pd.DataFrame.from_dict(anomaly, orient="index").rename_axis("Column"), use_container_width=True)
            else:
                st.info("No eligible numeric columns or no IQR outlier candidates were found.")
        if "create_cleaning_preview" in observations and "error" not in observations["create_cleaning_preview"]:
            st.subheader("Non-destructive cleaning recommendations")
            st.json(observations["create_cleaning_preview"])
        st.subheader("Ask the agent a follow-up question")
        question = st.text_input("Question", key="followup_question")
        if st.button("Ask Qwen3"):
            with st.spinner("Preparing a grounded answer..."):
                agent = DataQualityAgent(model=model, ollama_url=ollama_url)
                st.write(agent.answer_question(question, result))
        st.subheader("Feedback and adaptation")
        with st.form("feedback_form"):
            rating = st.slider("How useful was this investigation?", 1, 5, 4)
            category = st.selectbox("Most useful area", ["missing_values", "duplicates", "outliers", "cleaning", "general"])
            comment = st.text_area("What should the agent improve?")
            submitted = st.form_submit_button("Submit feedback")
            if submitted:
                agent = DataQualityAgent(model=model, ollama_url=ollama_url)
                agent.record_feedback(rating, comment, category)
                st.success("Feedback stored in SQLite memory and available to guide future investigations.")

st.markdown("---")
st.caption("Responsible-use note: outliers are candidates, not proof of errors. Review recommendations before changing source data.")
