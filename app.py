
"""
AI-Powered Data Quality Detective
Modern lavender dashboard with Qwen3, Streamlit and SQLite memory.
"""

import io
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from agent import DataQualityAgent


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Data Quality Detective",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. CUSTOM DESIGN
# =========================================================

st.markdown("""
<style>

/* Main application background */
.stApp {
    background: #F7F5FC;
    color: #29223D;
}

/* Main content spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* General text */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

p, label, li, .stMarkdown {
    color: #49415D;
}

/* Headings */
h1, h2, h3, h4 {
    color: #30234D !important;
    font-weight: 700 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #30234D;
    border-right: 1px solid #473866;
}

[data-testid="stSidebar"] * {
    color: #F5F0FF !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #E0D2FF !important;
}

/* Sidebar input fields */
[data-testid="stSidebar"] input {
    background: #45365F !important;
    color: white !important;
    border: 1px solid #76649B !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #45365F !important;
    border-color: #76649B !important;
}

/* Main input fields */
.stTextInput input,
.stTextArea textarea {
    background: #FFFFFF !important;
    color: #30234D !important;
    border: 1px solid #DCD3ED !important;
    border-radius: 10px !important;
}

/* Select boxes */
[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    border-radius: 10px !important;
    border-color: #DCD3ED !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 2px dashed #BCA7E8;
    border-radius: 16px;
    padding: 15px;
}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button {
    background: #7451D9 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.2rem !important;
    font-weight: 600 !important;
    transition: 0.2s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #5937B5 !important;
    border: none !important;
    color: #FFFFFF !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E8E0F5;
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 3px 12px rgba(48, 35, 77, 0.05);
}

[data-testid="stMetricLabel"] {
    color: #756B89 !important;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #5937B5 !important;
    font-weight: 750;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    border-bottom: 1px solid #E4DCEE;
}

.stTabs [data-baseweb="tab"] {
    background: #EEE9F7;
    color: #594B70;
    border-radius: 9px 9px 0 0;
    padding: 12px 18px;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background: #7451D9 !important;
    color: white !important;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid #E6DFF0;
    border-radius: 12px;
    overflow: hidden;
}

/* Expanders */
[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #E7E0F1;
    border-radius: 12px;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Custom hero */
.hero {
    background: linear-gradient(120deg, #30234D, #6043A5);
    padding: 32px;
    border-radius: 20px;
    margin-bottom: 25px;
    box-shadow: 0 8px 24px rgba(48, 35, 77, 0.15);
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: 32px;
    margin-bottom: 8px;
}

.hero p {
    color: #E6DDF8 !important;
    font-size: 15px;
    margin-bottom: 0;
}

.hero-tag {
    color: #E4D4FF;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 10px;
}

/* Section labels */
.section-label {
    color: #8065BC;
    font-size: 12px;
    font-weight: 750;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 6px;
}

/* Information cards */
.info-card {
    background: #FFFFFF;
    border: 1px solid #E8E0F5;
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 12px;
    box-shadow: 0 3px 12px rgba(48, 35, 77, 0.04);
}

.info-card h4 {
    margin-top: 0;
    margin-bottom: 8px;
}

.info-card p {
    margin-bottom: 0;
    font-size: 14px;
}

/* Footer */
.footer {
    text-align: center;
    color: #887E9B;
    font-size: 12px;
    padding: 25px 0 10px;
}

hr {
    border-color: #E8E0F5;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. SESSION STATE
# =========================================================

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

if "latest_answer" not in st.session_state:
    st.session_state.latest_answer = None

if "dataset_key" not in st.session_state:
    st.session_state.dataset_key = None

if "agent_model" not in st.session_state:
    st.session_state.agent_model = "qwen3:8b"

if "agent_url" not in st.session_state:
    st.session_state.agent_url = "http://localhost:11434"


# =========================================================
# 4. SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("""
    <div style="text-align:center; padding:12px 0 20px;">
        <div style="font-size:45px;">🔎</div>
        <h2 style="margin-bottom:3px;">Data Detective</h2>
        <p style="font-size:12px; color:#D2C5E8;">
            Intelligent Data Quality Assistant
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### ⚙️ Agent Configuration")

    model = st.text_input(
        "Ollama model",
        value="qwen3:8b",
        help="Enter your locally installed Ollama model."
    )

    ollama_url = st.text_input(
        "Ollama server URL",
        value="http://localhost:11434"
    )

    st.markdown("---")

    st.markdown("### 🧠 Agent Capabilities")

    st.markdown("""
    **01 · Intelligent Planning**

    Qwen3 plans investigation steps.

    **02 · Data Profiling**

    Examines dataset structure and statistics.

    **03 · Quality Detection**

    Identifies missing values and duplicates.

    **04 · Anomaly Detection**

    Finds potential numerical outliers.

    **05 · SQLite Memory**

    Stores investigation feedback and history.

    **06 · Adaptive Feedback**

    Uses feedback to guide future investigations.
    """)

    st.markdown("---")

    st.markdown("### 🛠️ Technology")

    st.markdown("""
    - Python
    - Streamlit
    - Ollama + Qwen3
    - Pandas
    - Matplotlib
    - SQLite
    """)

    st.markdown("---")

    st.caption("DAT404 · AI Agent Project")


# =========================================================
# 5. MAIN HEADER
# =========================================================

st.markdown("""
<div class="hero">
    <div class="hero-tag">AI-powered analytics platform</div>
    <h1>🔎 Data Quality Detective</h1>
    <p>
        Discover data quality issues, identify potential anomalies,
        and receive intelligent recommendations powered by Qwen3.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# 6. UPLOAD DATASET
# =========================================================

st.markdown(
    '<div class="section-label">01 / Data input</div>',
    unsafe_allow_html=True
)

st.subheader("Upload your dataset")

st.write(
    "Upload a CSV or Excel file to begin your intelligent data investigation."
)

uploaded = st.file_uploader(
    "Choose a CSV or Excel file",
    type=["csv", "xlsx", "xls"],
    help="Supported formats: CSV, XLSX and XLS."
)

df = None
selected_sheet = "CSV"

if uploaded is not None:

    try:

        file_bytes = uploaded.getvalue()

        if uploaded.name.lower().endswith(".csv"):

            df = pd.read_csv(io.BytesIO(file_bytes))

        else:

            excel_file = pd.ExcelFile(io.BytesIO(file_bytes))

            selected_sheet = st.selectbox(
                "📑 Select worksheet",
                excel_file.sheet_names
            )

            df = pd.read_excel(
                io.BytesIO(file_bytes),
                sheet_name=selected_sheet
            )

        # Detect when a new dataset or sheet is selected
        current_key = (
            uploaded.name,
            len(file_bytes),
            selected_sheet
        )

        if current_key != st.session_state.dataset_key:

            st.session_state.latest_result = None
            st.session_state.latest_answer = None
            st.session_state.dataset_key = current_key

        st.success(
            f"Successfully loaded: {uploaded.name}"
        )

    except Exception as e:

        st.error(f"Unable to read the uploaded file: {e}")


# =========================================================
# 7. DATASET OVERVIEW
# =========================================================

if df is not None:

    st.markdown("---")

    st.markdown(
        '<div class="section-label">02 / Dataset overview</div>',
        unsafe_allow_html=True
    )

    st.subheader("Your dataset at a glance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total rows",
        f"{len(df):,}"
    )

    col2.metric(
        "Total columns",
        f"{len(df.columns):,}"
    )

    col3.metric(
        "Missing cells",
        f"{int(df.isna().sum().sum()):,}"
    )

    col4.metric(
        "Duplicate rows",
        f"{int(df.duplicated().sum()):,}"
    )

    # Dataset preview
    with st.expander("👀 Preview uploaded dataset", expanded=True):

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

        st.caption(
            f"Showing the first {min(10, len(df))} rows "
            f"out of {len(df):,} total rows."
        )


# =========================================================
# 8. INVESTIGATION SECTION
# =========================================================

if df is not None:

    st.markdown("---")

    st.markdown(
        '<div class="section-label">03 / Intelligent investigation</div>',
        unsafe_allow_html=True
    )

    st.subheader("What should the agent investigate?")

    goal = st.text_area(
        "Investigation goal",
        value=(
            "Investigate this dataset, identify data quality issues "
            "and numeric outlier candidates, and recommend safe next steps."
        ),
        height=110
    )

    run_button = st.button(
        "🚀  Run intelligent investigation",
        type="primary",
        use_container_width=True
    )

    if run_button:

        try:

            with st.spinner(
                "Qwen3 is planning and the agent is executing its tools..."
            ):

                agent = DataQualityAgent(
                    model=model,
                    ollama_url=ollama_url
                )

                result = agent.investigate(df, goal)

                st.session_state.latest_result = result
                st.session_state.latest_answer = None
                st.session_state.agent_model = model
                st.session_state.agent_url = ollama_url

            st.success(
                "Investigation completed successfully!"
            )

        except Exception as e:

            st.error(
                f"Investigation could not be completed: {e}"
            )


# =========================================================
# 9. DISPLAY INVESTIGATION RESULTS
# =========================================================

result = st.session_state.latest_result

if result is not None:

    st.markdown("---")

    st.markdown(
        '<div class="section-label">04 / Investigation results</div>',
        unsafe_allow_html=True
    )

    st.subheader("Your investigation report")

    st.caption(
        f"Model: {st.session_state.agent_model}  |  "
        f"Execution time: "
        f"{result.get('latency_seconds', 'N/A')} seconds"
    )

    # Tabs for organised presentation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Findings",
        "📋 Quality details",
        "⚙️ Execution trace",
        "💬 Ask the agent",
        "⭐ Feedback & memory"
    ])

    observations = result.get("observations", {})

    # -----------------------------------------------------
    # TAB 1: AGENT FINDINGS
    # -----------------------------------------------------

    with tab1:

        st.markdown("### 🧠 Agent explanation")

        st.markdown(
            f"""
            <div class="info-card">
                <h4>Investigation summary</h4>
                <p>{result.get('summary', 'No explanation available.')}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        if (
            "profile_dataset" in observations
            and "error" not in observations["profile_dataset"]
        ):

            profile = observations["profile_dataset"]

            st.markdown("### Dataset profile")

            p1, p2 = st.columns(2)

            p1.metric(
                "Rows analysed",
                f"{profile.get('rows', 0):,}"
            )

            p2.metric(
                "Columns analysed",
                f"{profile.get('columns', 0):,}"
            )

        if (
            "detect_quality_issues" in observations
            and "error" not in observations["detect_quality_issues"]
        ):

            quality = observations["detect_quality_issues"]

            st.markdown("### Data quality overview")

            q1, q2, q3 = st.columns(3)

            q1.metric(
                "Missing cells",
                quality.get("total_missing_cells", 0)
            )

            q2.metric(
                "Duplicate rows",
                quality.get("duplicate_rows", 0)
            )

            q3.metric(
                "Constant columns",
                len(quality.get("constant_columns", []))
            )

        if (
            "detect_anomalies" in observations
            and "error" not in observations["detect_anomalies"]
        ):

            anomaly = observations["detect_anomalies"].get(
                "columns", {}
            )

            st.markdown("### Potential numerical outliers")

            if anomaly:

                chart_df = pd.DataFrame({
                    "Column": list(anomaly.keys()),
                    "Candidates": [
                        v.get("candidate_count", 0)
                        for v in anomaly.values()
                    ]
                })

                fig, ax = plt.subplots(figsize=(9, 4))

                bars = ax.bar(
                    chart_df["Column"],
                    chart_df["Candidates"],
                    color="#8061D6",
                    width=0.55
                )

                ax.set_title(
                    "Potential Outliers by Column",
                    fontsize=14,
                    color="#30234D",
                    pad=15
                )

                ax.set_ylabel("Candidate Count")
                ax.set_xlabel("Dataset Columns")

                ax.set_facecolor("#FFFFFF")
                fig.patch.set_facecolor("#FFFFFF")

                ax.bar_label(bars, padding=4)

                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                plt.tight_layout()

                st.pyplot(fig)
                plt.close(fig)

            else:

                st.info(
                    "No eligible numeric columns or no IQR "
                    "outlier candidates were found."
                )

    # -----------------------------------------------------
    # TAB 2: DATA QUALITY DETAILS
    # -----------------------------------------------------

    with tab2:

        st.markdown("### 🔍 Detailed quality findings")

        if (
            "detect_quality_issues" in observations
            and "error" not in observations["detect_quality_issues"]
        ):

            quality = observations["detect_quality_issues"]

            missing_values = quality.get("missing_values", {})

            st.markdown("#### Missing-value analysis")

            if missing_values:

                missing_df = pd.DataFrame.from_dict(
                    missing_values,
                    orient="index"
                ).rename_axis("Column")

                st.dataframe(
                    missing_df,
                    use_container_width=True
                )

            else:

                st.success("No missing values were detected.")

            st.markdown("#### Duplicate and constant columns")

            st.write(
                "**Duplicate rows:**",
                quality.get("duplicate_rows", 0)
            )

            st.write(
                "**Constant columns:**",
                quality.get("constant_columns", [])
            )

        else:

            st.info(
                "Data quality details are not available for this investigation."
            )

        st.markdown("---")

        st.markdown("#### Potential outlier details")

        if (
            "detect_anomalies" in observations
            and "error" not in observations["detect_anomalies"]
        ):

            anomaly = observations["detect_anomalies"].get(
                "columns", {}
            )

            if anomaly:

                anomaly_df = pd.DataFrame.from_dict(
                    anomaly,
                    orient="index"
                ).rename_axis("Column")

                st.dataframe(
                    anomaly_df,
                    use_container_width=True
                )

            else:

                st.info("No eligible outlier results are available.")

        # Cleaning preview
        st.markdown("---")

        st.markdown("#### 🧹 Non-destructive cleaning recommendations")

        if (
            "create_cleaning_preview" in observations
            and "error" not in observations["create_cleaning_preview"]
        ):

            with st.expander(
                "View cleaning recommendations",
                expanded=True
            ):

                st.json(
                    observations["create_cleaning_preview"]
                )

            st.caption(
                "These are recommendations only. Your original dataset "
                "is not modified automatically."
            )

        else:

            st.info("No cleaning preview is available.")

    # -----------------------------------------------------
    # TAB 3: EXECUTION TRACE
    # -----------------------------------------------------

    with tab3:

        st.markdown("### ⚙️ Agent execution trace")

        st.write(
            "This section shows the steps taken by the agent "
            "during the investigation."
        )

        trace = result.get("trace", [])

        if trace:

            trace_df = pd.DataFrame(trace)

            st.dataframe(
                trace_df,
                use_container_width=True
            )

            st.metric(
                "Total execution steps",
                len(trace)
            )

        else:

            st.info("No execution trace is available.")

        st.markdown("---")

        st.markdown("#### Agent architecture")

        st.markdown("""
        <div class="info-card">
            <p><b>1. Perceive</b> — Receives the uploaded dataset.</p>
            <p><b>2. Plan</b> — Qwen3 selects appropriate tools.</p>
            <p><b>3. Validate</b> — Checks the planned tool actions.</p>
            <p><b>4. Act</b> — Executes Python analysis tools.</p>
            <p><b>5. Observe</b> — Collects tool results.</p>
            <p><b>6. Explain</b> — Produces evidence-grounded findings.</p>
            <p><b>7. Remember</b> — Stores feedback in SQLite memory.</p>
        </div>
        """, unsafe_allow_html=True)

    # -----------------------------------------------------
    # TAB 4: FOLLOW-UP QUESTIONS
    # -----------------------------------------------------

    with tab4:

        st.markdown("### 💬 Ask Qwen3 a follow-up question")

        st.write(
            "Ask questions about the findings from your current investigation."
        )

        with st.form("followup_form"):

            question = st.text_area(
                "Your question",
                placeholder=(
                    "Example: Which column has the most missing values?"
                ),
                height=100
            )

            ask_button = st.form_submit_button(
                "Ask the agent",
                use_container_width=True
            )

            if ask_button:

                if not question.strip():

                    st.warning("Please enter a question first.")

                else:

                    try:

                        with st.spinner(
                            "Qwen3 is preparing an answer..."
                        ):

                            agent = DataQualityAgent(
                                model=st.session_state.agent_model,
                                ollama_url=st.session_state.agent_url
                            )

                            answer = agent.answer_question(
                                question,
                                result
                            )

                            st.session_state.latest_answer = answer

                    except Exception as e:

                        st.error(f"Unable to answer: {e}")

        if st.session_state.latest_answer is not None:

            st.markdown("#### Agent response")

            st.markdown(
                f"""
                <div class="info-card">
                    <p>{st.session_state.latest_answer}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    # -----------------------------------------------------
    # TAB 5: FEEDBACK AND MEMORY
    # -----------------------------------------------------

    with tab5:

        st.markdown("### ⭐ Help the agent improve")

        st.write(
            "Rate the investigation and provide feedback. "
            "Your feedback is stored in SQLite memory and can "
            "guide future investigations."
        )

        with st.form("feedback_form"):

            rating = st.slider(
                "How useful was this investigation?",
                min_value=1,
                max_value=5,
                value=4
            )

            category = st.selectbox(
                "Which area was most useful?",
                [
                    "missing_values",
                    "duplicates",
                    "outliers",
                    "cleaning",
                    "general"
                ]
            )

            comment = st.text_area(
                "Your feedback",
                placeholder=(
                    "Tell the agent what was useful or what it should improve."
                )
            )

            feedback_submit = st.form_submit_button(
                "⭐ Submit feedback",
                use_container_width=True
            )

            if feedback_submit:

                try:

                    agent = DataQualityAgent(
                        model=st.session_state.agent_model,
                        ollama_url=st.session_state.agent_url
                    )

                    agent.record_feedback(
                        rating,
                        comment,
                        category
                    )

                    st.success(
                        "Feedback successfully stored in SQLite memory!"
                    )

                except Exception as e:

                    st.error(
                        f"Could not store feedback: {e}"
                    )

        st.markdown("---")

        st.markdown("#### How the memory works")

        st.markdown("""
        <div class="info-card">
            <h4>🧠 SQLite Episodic Memory</h4>
            <p>
                The agent stores user feedback and investigation
                information in a local SQLite database.
                Feedback summaries can influence the priorities
                of future investigations.
            </p>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# 10. EMPTY STATE
# =========================================================

else:

    st.markdown("---")

    st.markdown(
        '<div class="section-label">Getting started</div>',
        unsafe_allow_html=True
    )

    st.subheader("Your intelligent data assistant is ready.")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown("""
        <div class="info-card">
            <h4>📁 Upload data</h4>
            <p>
                Import CSV or Excel datasets
                for intelligent investigation.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown("""
        <div class="info-card">
            <h4>🔍 Detect issues</h4>
            <p>
                Identify missing values, duplicates
                and potential numerical outliers.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c3:

        st.markdown("""
        <div class="info-card">
            <h4>💡 Get insights</h4>
            <p>
                Receive explanations and safe
                data-cleaning recommendations.
            </p>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# 11. FOOTER
# =========================================================

st.markdown("---")

st.markdown("""
<div class="footer">
    <b>AI-Powered Data Quality Detective</b><br>
    DAT404 · Intelligent Agent Project<br><br>
    Responsible use: Outliers are candidates, not proof of errors.
    Always review recommendations before modifying source data.
</div>
""", unsafe_allow_html=True)
