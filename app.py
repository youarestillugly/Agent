
"""
AI-Powered Data Quality Detective
LangGraph + Qwen3 + Streamlit + SQLite Memory
Data Cleaning Studio with visual comparisons and downloads.
"""

import io
import hashlib

import numpy as np
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

.stApp {
    background: #F7F5FC;
    color: #29223D;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

p, label, li, .stMarkdown {
    color: #49415D;
}

h1, h2, h3, h4 {
    color: #30234D !important;
    font-weight: 700 !important;
}

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

[data-testid="stSidebar"] input {
    background: #45365F !important;
    color: white !important;
    border: 1px solid #76649B !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #45365F !important;
    border-color: #76649B !important;
}

.stTextInput input,
.stTextArea textarea {
    background: #FFFFFF !important;
    color: #30234D !important;
    border: 1px solid #DCD3ED !important;
    border-radius: 10px !important;
}

[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    border-radius: 10px !important;
    border-color: #DCD3ED !important;
}

.stSlider [data-baseweb="slider"] {
    color: #7451D9;
}

[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 2px dashed #BCA7E8;
    border-radius: 16px;
    padding: 15px;
}

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
    color: #FFFFFF !important;
}

[data-testid="stDownloadButton"] > button {
    background: #5937B5 !important;
    color: white !important;
    border-radius: 10px !important;
    border: none !important;
    font-weight: 600 !important;
}

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

[data-testid="stDataFrame"] {
    border: 1px solid #E6DFF0;
    border-radius: 12px;
    overflow: hidden;
}

[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #E7E0F1;
    border-radius: 12px;
}

[data-testid="stAlert"] {
    border-radius: 12px;
}

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

.section-label {
    color: #8065BC;
    font-size: 12px;
    font-weight: 750;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 6px;
}

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

DEFAULTS = {
    "latest_result": None,
    "latest_answer": None,
    "dataset_key": None,
    "agent_model": "qwen3:8b",
    "agent_url": "http://localhost:11434",
    "last_feedback": None,
    "cleaned_df": None,
    "cleaning_log": [],
    "cleaning_dataset_key": None
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# 4. HELPER FUNCTIONS
# =========================================================

def create_agent():
    """Create the existing LangGraph-based agent."""

    return DataQualityAgent(
        model=st.session_state.agent_model,
        ollama_url=st.session_state.agent_url
    )


def show_metric_row(items):
    """Display multiple metrics in one row."""

    columns = st.columns(len(items))

    for column, (label, value) in zip(columns, items):
        column.metric(label, value)


def show_observation_error(observation):
    """Check whether a tool observation contains an error."""

    return (
        not isinstance(observation, dict)
        or "error" in observation
    )


def clean_dataset(
    original_df,
    handle_missing,
    missing_strategy,
    remove_duplicates,
    handle_outliers,
    outlier_columns,
    outlier_strategy
):
    """
    Clean a copy of the original dataset.

    Returns:
        cleaned_df
        cleaning_records
    """

    cleaned = original_df.copy(deep=True)
    records = []

    # -----------------------------------------------------
    # 1. MISSING VALUE HANDLING
    # -----------------------------------------------------

    if handle_missing:

        total_filled = 0
        columns_affected = []

        if missing_strategy == "Remove rows with missing values":

            before = len(cleaned)
            cleaned = cleaned.dropna().copy()
            removed = before - len(cleaned)

            records.append({
                "Operation": "Missing values",
                "Method": "Remove rows with missing values",
                "Affected records": removed,
                "Details": f"{removed} rows removed"
            })

        else:

            for column in cleaned.columns:

                missing_count = int(cleaned[column].isna().sum())

                if missing_count == 0:
                    continue

                if pd.api.types.is_numeric_dtype(cleaned[column]):

                    if missing_strategy == "Fill with median / mode":
                        fill_value = cleaned[column].median()
                    else:
                        fill_value = cleaned[column].mean()

                else:

                    mode_values = cleaned[column].mode(
                        dropna=True
                    )

                    if len(mode_values) > 0:
                        fill_value = mode_values.iloc[0]
                    else:
                        continue

                if pd.notna(fill_value):

                    cleaned[column] = cleaned[column].fillna(
                        fill_value
                    )

                    total_filled += missing_count
                    columns_affected.append(str(column))

            records.append({
                "Operation": "Missing values",
                "Method": missing_strategy,
                "Affected records": total_filled,
                "Details": (
                    f"{total_filled} missing cells filled. "
                    f"Columns: {', '.join(columns_affected)}"
                )
            })

    # -----------------------------------------------------
    # 2. DUPLICATE REMOVAL
    # -----------------------------------------------------

    if remove_duplicates:

        before = len(cleaned)

        cleaned = cleaned.drop_duplicates(
            keep="first"
        ).copy()

        removed = before - len(cleaned)

        records.append({
            "Operation": "Duplicate records",
            "Method": "Keep first occurrence",
            "Affected records": removed,
            "Details": f"{removed} duplicate rows removed"
        })

    # -----------------------------------------------------
    # 3. NUMERICAL OUTLIER HANDLING
    # -----------------------------------------------------

    if handle_outliers:

        total_candidates = 0
        total_removed = 0
        columns_processed = []

        rows_to_remove = pd.Series(
            False,
            index=cleaned.index
        )

        for column in outlier_columns:

            if column not in cleaned.columns:
                continue

            values = pd.to_numeric(
                cleaned[column],
                errors="coerce"
            ).dropna()

            if len(values) < 4:
                continue

            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)

            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outlier_mask = (
                (cleaned[column] < lower_bound)
                | (cleaned[column] > upper_bound)
            ).fillna(False)

            candidate_count = int(outlier_mask.sum())

            if candidate_count == 0:
                continue

            total_candidates += candidate_count
            columns_processed.append(str(column))

            if outlier_strategy == "Cap values using IQR bounds":

                cleaned[column] = cleaned[column].clip(
                    lower=lower_bound,
                    upper=upper_bound
                )

            else:

                rows_to_remove = (
                    rows_to_remove | outlier_mask
                )

        if (
            handle_outliers
            and outlier_strategy == "Remove rows containing IQR outliers"
        ):

            total_removed = int(rows_to_remove.sum())

            cleaned = cleaned.loc[
                ~rows_to_remove
            ].copy()

        records.append({
            "Operation": "Numerical outliers",
            "Method": outlier_strategy,
            "Affected records": (
                total_removed
                if outlier_strategy == "Remove rows containing IQR outliers"
                else total_candidates
            ),
            "Details": (
                f"{total_candidates} outlier candidates identified. "
                f"{total_removed} rows removed. "
                f"Columns: {', '.join(columns_processed)}"
            )
        })

    return cleaned, records


# =========================================================
# 5. SIDEBAR
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
        value=st.session_state.agent_model,
        help="Enter your locally installed Ollama model."
    )

    ollama_url = st.text_input(
        "Ollama server URL",
        value=st.session_state.agent_url
    )

    st.session_state.agent_model = model.strip()
    st.session_state.agent_url = ollama_url.strip()

    st.markdown("---")

    st.markdown("### 🧠 Agent Capabilities")

    st.markdown("""
    **01 · Intelligent Planning**

    Qwen3 helps plan investigation steps.

    **02 · LangGraph Workflow**

    Organises the agent into connected workflow nodes.

    **03 · Data Profiling**

    Examines dataset structure and statistics.

    **04 · Quality Detection**

    Identifies missing values and duplicates.

    **05 · Anomaly Detection**

    Finds potential numerical outliers.

    **06 · Data Cleaning**

    Applies user-selected cleaning operations.

    **07 · SQLite Memory**

    Stores investigation history and feedback.
    """)

    st.markdown("---")

    st.markdown("### 🛠️ Technology")

    st.markdown("""
    - Python
    - Streamlit
    - LangGraph
    - Ollama + Qwen3
    - Pandas
    - NumPy
    - Matplotlib
    - SQLite
    """)

    st.markdown("---")

    st.caption("DAT404 · AI Agent Project")


# =========================================================
# 6. MAIN HEADER
# =========================================================

st.markdown("""
<div class="hero">
    <div class="hero-tag">AI-powered analytics platform</div>
    <h1>🔎 Data Quality Detective</h1>
    <p>
        Discover data quality issues, identify potential anomalies,
        clean your dataset and visualise improvements using Qwen3
        and LangGraph.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# 7. UPLOAD DATASET
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

            df = pd.read_csv(
                io.BytesIO(file_bytes)
            )

        else:

            excel_file = pd.ExcelFile(
                io.BytesIO(file_bytes)
            )

            selected_sheet = st.selectbox(
                "📑 Select worksheet",
                excel_file.sheet_names,
                key="excel_sheet_selector"
            )

            df = pd.read_excel(
                io.BytesIO(file_bytes),
                sheet_name=selected_sheet
            )

        file_hash = hashlib.md5(
            file_bytes
        ).hexdigest()

        current_key = (
            uploaded.name,
            file_hash,
            selected_sheet
        )

        if current_key != st.session_state.dataset_key:

            st.session_state.latest_result = None
            st.session_state.latest_answer = None
            st.session_state.cleaned_df = None
            st.session_state.cleaning_log = []

            st.session_state.dataset_key = current_key
            st.session_state.cleaning_dataset_key = current_key

        st.success(
            f"Successfully loaded: {uploaded.name}"
        )

    except Exception as e:

        df = None

        st.error(
            f"Unable to read the uploaded file: {e}"
        )


# =========================================================
# 8. DATASET OVERVIEW
# =========================================================

if df is not None:

    st.markdown("---")

    st.markdown(
        '<div class="section-label">02 / Dataset overview</div>',
        unsafe_allow_html=True
    )

    st.subheader("Your dataset at a glance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total rows", f"{len(df):,}")
    col2.metric("Total columns", f"{len(df.columns):,}")
    col3.metric(
        "Missing cells",
        f"{int(df.isna().sum().sum()):,}"
    )
    col4.metric(
        "Duplicate rows",
        f"{int(df.duplicated().sum()):,}"
    )

    with st.expander(
        "👀 Preview uploaded dataset",
        expanded=True
    ):

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

        st.caption(
            f"Showing the first {min(10, len(df))} rows "
            f"out of {len(df):,} total rows."
        )


# =========================================================
# 9. INVESTIGATION SECTION
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
        height=110,
        key="investigation_goal"
    )

    run_button = st.button(
        "🚀  Run intelligent investigation",
        type="primary",
        use_container_width=True
    )

    if run_button:

        if not goal.strip():

            st.warning(
                "Please enter an investigation goal."
            )

        else:

            try:

                with st.spinner(
                    "LangGraph is coordinating the investigation workflow..."
                ):

                    agent = create_agent()

                    result = agent.investigate(
                        df.copy(),
                        goal
                    )

                    st.session_state.latest_result = result
                    st.session_state.latest_answer = None

                    st.session_state.cleaned_df = None
                    st.session_state.cleaning_log = []

                st.success(
                    "Investigation completed successfully!"
                )

            except Exception as e:

                st.error(
                    f"Investigation could not be completed: {e}"
                )

                st.info(
                    "Check that Ollama is running, the selected model "
                    "is installed, and the LangGraph dependencies "
                    "are available."
                )


# =========================================================
# 10. DISPLAY INVESTIGATION RESULTS
# =========================================================

result = st.session_state.latest_result

if result is not None:

    st.markdown("---")

    st.markdown(
        '<div class="section-label">04 / Investigation results</div>',
        unsafe_allow_html=True
    )

    st.subheader("Your investigation findings")

    latency = result.get(
        "latency_seconds",
        "N/A"
    )

    st.caption(
        f"Model: {st.session_state.agent_model}  |  "
        f"Execution time: {latency} seconds"
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Findings",
        "📋 Quality details",
        "⚙️ Execution trace",
        "💬 Ask the agent",
        "⭐ Feedback & memory"
    ])

    observations = result.get(
        "observations",
        {}
    )

    # -----------------------------------------------------
    # TAB 1: AGENT FINDINGS
    # -----------------------------------------------------

    with tab1:

        st.markdown("### 🧠 Agent explanation")

        st.markdown("#### Investigation summary")

        st.write(
            result.get(
                "summary",
                "No explanation available."
            )
        )

        if (
            "profile_dataset" in observations
            and not show_observation_error(
                observations["profile_dataset"]
            )
        ):

            profile = observations["profile_dataset"]

            st.markdown("### Dataset profile")

            show_metric_row([
                (
                    "Rows analysed",
                    f"{profile.get('rows', 0):,}"
                ),
                (
                    "Columns analysed",
                    f"{profile.get('columns', 0):,}"
                )
            ])

        if (
            "detect_quality_issues" in observations
            and not show_observation_error(
                observations["detect_quality_issues"]
            )
        ):

            quality = observations["detect_quality_issues"]

            st.markdown("### Data quality overview")

            show_metric_row([
                (
                    "Missing cells",
                    quality.get("total_missing_cells", 0)
                ),
                (
                    "Duplicate rows",
                    quality.get("duplicate_rows", 0)
                ),
                (
                    "Constant columns",
                    len(quality.get("constant_columns", []))
                )
            ])

        if (
            "detect_anomalies" in observations
            and not show_observation_error(
                observations["detect_anomalies"]
            )
        ):

            anomaly = observations[
                "detect_anomalies"
            ].get("columns", {})

            st.markdown("### Potential numerical outliers")

            if anomaly:

                chart_df = pd.DataFrame({
                    "Column": list(anomaly.keys()),
                    "Candidates": [
                        value.get("candidate_count", 0)
                        for value in anomaly.values()
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

                ax.bar_label(
                    bars,
                    padding=4
                )

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

        # =================================================
        # DATA CLEANING STUDIO
        # THIS SECTION APPEARS ONLY UNDER FINDINGS
        # =================================================

        st.markdown("---")

        st.markdown(
            '<div class="section-label">05 / Data cleaning studio</div>',
            unsafe_allow_html=True
        )

        st.subheader("🧹 Clean your dataset")

        st.write(
            "Review the investigation findings, select the cleaning "
            "operations you want, and generate an improved dataset."
        )

        original_df = df.copy(deep=True)

        # Reset cleaning results when dataset changes
        if (
            st.session_state.cleaning_dataset_key
            != st.session_state.dataset_key
        ):

            st.session_state.cleaned_df = None
            st.session_state.cleaning_log = []

            st.session_state.cleaning_dataset_key = (
                st.session_state.dataset_key
            )

        # -------------------------------------------------
        # CLEANING SETTINGS
        # -------------------------------------------------

        st.markdown("### 1. Select cleaning operations")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("#### Missing values")

            handle_missing = st.checkbox(
                "Handle missing values",
                value=True,
                key="handle_missing"
            )

            missing_strategy = st.selectbox(
                "Missing-value strategy",
                [
                    "Fill with median / mode",
                    "Fill with mean / mode",
                    "Remove rows with missing values"
                ],
                key="missing_strategy"
            )

        with col2:

            st.markdown("#### Duplicate records")

            remove_duplicates = st.checkbox(
                "Remove duplicate rows",
                value=True,
                key="remove_duplicates"
            )

            st.caption(
                "Removes duplicate rows while keeping the first occurrence."
            )

        st.markdown("---")

        st.markdown("#### Numerical outliers")

        numeric_columns = original_df.select_dtypes(
            include=np.number
        ).columns.tolist()

        handle_outliers = st.checkbox(
            "Handle potential numerical outliers",
            value=False,
            key="handle_outliers"
        )

        if numeric_columns:

            outlier_columns = st.multiselect(
                "Select numerical columns",
                numeric_columns,
                default=[],
                key="outlier_columns"
            )

        else:

            outlier_columns = []

            st.info(
                "This dataset does not contain numerical columns."
            )

        outlier_strategy = st.selectbox(
            "Outlier treatment",
            [
                "Cap values using IQR bounds",
                "Remove rows containing IQR outliers"
            ],
            key="outlier_strategy"
        )

        st.caption(
            "IQR uses Q1 − 1.5 × IQR and Q3 + 1.5 × IQR "
            "to identify potential outliers."
        )

        # -------------------------------------------------
        # APPLY CLEANING
        # -------------------------------------------------

        st.markdown("---")

        apply_cleaning = st.button(
            "🧹 Apply selected cleaning operations",
            type="primary",
            use_container_width=True,
            key="apply_cleaning_button"
        )

        if apply_cleaning:

            try:

                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.info(
                    "Preparing a separate copy of your dataset..."
                )

                progress_bar.progress(15)

                status_text.info(
                    "Applying missing-value treatment..."
                )

                progress_bar.progress(35)

                status_text.info(
                    "Checking and removing duplicate records..."
                )

                progress_bar.progress(55)

                status_text.info(
                    "Checking numerical outliers..."
                )

                cleaned, records = clean_dataset(
                    original_df=original_df,
                    handle_missing=handle_missing,
                    missing_strategy=missing_strategy,
                    remove_duplicates=remove_duplicates,
                    handle_outliers=handle_outliers,
                    outlier_columns=outlier_columns,
                    outlier_strategy=outlier_strategy
                )

                progress_bar.progress(90)

                status_text.info(
                    "Saving the cleaned dataset and cleaning records..."
                )

                st.session_state.cleaned_df = cleaned.copy(deep=True)
                st.session_state.cleaning_log = records

                progress_bar.progress(100)

                status_text.success(
                    "Cleaning completed successfully!"
                )

            except Exception as e:

                st.error(
                    f"Unable to clean the dataset: {e}"
                )

        # =================================================
        # CLEANING RESULTS
        # =================================================

        if st.session_state.cleaned_df is not None:

            cleaned_df = st.session_state.cleaned_df

            st.markdown("---")

            st.markdown("### 2. Cleaning results")

            original_rows = len(original_df)
            cleaned_rows = len(cleaned_df)

            original_missing = int(
                original_df.isna().sum().sum()
            )

            cleaned_missing = int(
                cleaned_df.isna().sum().sum()
            )

            original_duplicates = int(
                original_df.duplicated().sum()
            )

            cleaned_duplicates = int(
                cleaned_df.duplicated().sum()
            )

            m1, m2, m3, m4 = st.columns(4)

            m1.metric(
                "Original rows",
                f"{original_rows:,}"
            )

            m2.metric(
                "Cleaned rows",
                f"{cleaned_rows:,}",
                delta=f"{cleaned_rows - original_rows:,}"
            )

            m3.metric(
                "Missing cells remaining",
                f"{cleaned_missing:,}",
                delta=f"{cleaned_missing - original_missing:,}"
            )

            m4.metric(
                "Duplicates remaining",
                f"{cleaned_duplicates:,}",
                delta=f"{cleaned_duplicates - original_duplicates:,}"
            )

            # ---------------------------------------------
            # CLEANING RECORDS
            # ---------------------------------------------

            st.markdown("### 3. Cleaning records")

            if st.session_state.cleaning_log:

                cleaning_log_df = pd.DataFrame(
                    st.session_state.cleaning_log
                )

                st.dataframe(
                    cleaning_log_df,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.info(
                    "No cleaning operations were selected."
                )

            # ---------------------------------------------
            # CLEANED DATASET PREVIEW
            # ---------------------------------------------

            st.markdown("---")

            st.markdown(
                "### 4. Cleaned dataset preview"
            )

            st.dataframe(
                cleaned_df.head(20),
                use_container_width=True
            )

            st.caption(
                f"Displaying {min(20, len(cleaned_df))} of "
                f"{len(cleaned_df):,} cleaned rows."
            )

            # =============================================
            # VISUAL COMPARISONS
            # =============================================

            st.markdown("---")

            st.markdown(
                "### 5. Visual comparison"
            )

            # ---------------------------------------------
            # MISSING VALUES GRAPH
            # ---------------------------------------------

            st.markdown(
                "#### Missing values before and after cleaning"
            )

            before_missing = original_df.isna().sum()
            after_missing = cleaned_df.isna().sum()

            missing_comparison = pd.DataFrame({
                "Before cleaning": before_missing,
                "After cleaning": after_missing
            })

            missing_comparison = missing_comparison[
                (
                    missing_comparison["Before cleaning"] > 0
                ) |
                (
                    missing_comparison["After cleaning"] > 0
                )
            ]

            if not missing_comparison.empty:

                fig, ax = plt.subplots(figsize=(10, 4))

                missing_comparison.plot(
                    kind="bar",
                    ax=ax,
                    color=["#BCA7E8", "#7451D9"],
                    width=0.7
                )

                ax.set_title(
                    "Missing Values: Before vs After",
                    color="#30234D",
                    fontsize=14
                )

                ax.set_xlabel("Columns")
                ax.set_ylabel("Missing value count")

                ax.set_facecolor("#FFFFFF")
                fig.patch.set_facecolor("#FFFFFF")

                ax.legend(frameon=False)

                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                plt.xticks(
                    rotation=45,
                    ha="right"
                )

                plt.tight_layout()

                st.pyplot(fig)
                plt.close(fig)

            else:

                st.success(
                    "There are no missing values before or after cleaning."
                )

            # ---------------------------------------------
            # DUPLICATE COMPARISON GRAPH
            # ---------------------------------------------

            st.markdown(
                "#### Duplicate rows comparison"
            )

            duplicate_comparison = pd.DataFrame({
                "Dataset": [
                    "Before cleaning",
                    "After cleaning"
                ],
                "Duplicate rows": [
                    original_duplicates,
                    cleaned_duplicates
                ]
            })

            fig, ax = plt.subplots(figsize=(7, 3))

            bars = ax.bar(
                duplicate_comparison["Dataset"],
                duplicate_comparison["Duplicate rows"],
                color=["#BCA7E8", "#7451D9"],
                width=0.5
            )

            ax.set_title(
                "Duplicate Rows: Before vs After",
                color="#30234D"
            )

            ax.set_ylabel("Duplicate count")

            ax.bar_label(
                bars,
                padding=3
            )

            ax.set_facecolor("#FFFFFF")
            fig.patch.set_facecolor("#FFFFFF")

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            plt.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

            # ---------------------------------------------
            # NUMERICAL DISTRIBUTION GRAPH
            # ---------------------------------------------

            st.markdown(
                "#### Numerical distribution comparison"
            )

            comparable_columns = [
                column
                for column in numeric_columns
                if column in cleaned_df.columns
            ]

            if comparable_columns:

                selected_graph_column = st.selectbox(
                    "Select numerical column",
                    comparable_columns,
                    key="cleaned_graph_column"
                )

                before_values = pd.to_numeric(
                    original_df[selected_graph_column],
                    errors="coerce"
                ).dropna()

                after_values = pd.to_numeric(
                    cleaned_df[selected_graph_column],
                    errors="coerce"
                ).dropna()

                fig, ax = plt.subplots(figsize=(10, 4))

                if not before_values.empty:

                    ax.hist(
                        before_values,
                        bins=20,
                        alpha=0.5,
                        label="Before cleaning",
                        color="#BCA7E8"
                    )

                if not after_values.empty:

                    ax.hist(
                        after_values,
                        bins=20,
                        alpha=0.6,
                        label="After cleaning",
                        color="#7451D9"
                    )

                ax.set_title(
                    f"{selected_graph_column}: Before vs After",
                    color="#30234D"
                )

                ax.set_xlabel(selected_graph_column)
                ax.set_ylabel("Frequency")

                ax.legend(frameon=False)

                ax.set_facecolor("#FFFFFF")
                fig.patch.set_facecolor("#FFFFFF")

                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                plt.tight_layout()

                st.pyplot(fig)
                plt.close(fig)

                # -----------------------------------------
                # STATISTICAL COMPARISON
                # -----------------------------------------

                st.markdown(
                    "#### Statistical comparison"
                )

                statistics = pd.DataFrame({
                    "Before cleaning": original_df[
                        selected_graph_column
                    ].describe(),

                    "After cleaning": cleaned_df[
                        selected_graph_column
                    ].describe()
                })

                st.dataframe(
                    statistics,
                    use_container_width=True
                )

            else:

                st.info(
                    "No numerical columns are available for distribution graphs."
                )

            # =============================================
            # DOWNLOAD CLEANED DATASET
            # =============================================

            st.markdown("---")

            st.markdown(
                "### 6. Download cleaned dataset"
            )

            st.write(
                "Your cleaned dataset is ready. "
                "Choose CSV or Excel format."
            )

            download_col1, download_col2 = st.columns(2)

            # ---------------------------------------------
            # CSV DOWNLOAD
            # ---------------------------------------------

            csv_data = cleaned_df.to_csv(
                index=False
            ).encode("utf-8-sig")

            with download_col1:

                st.download_button(
                    label="⬇️ Download cleaned CSV",
                    data=csv_data,
                    file_name="cleaned_dataset.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_cleaned_csv"
                )

            # ---------------------------------------------
            # EXCEL DOWNLOAD
            # ---------------------------------------------

            excel_buffer = io.BytesIO()

            with pd.ExcelWriter(
                excel_buffer,
                engine="openpyxl"
            ) as writer:

                cleaned_df.to_excel(
                    writer,
                    sheet_name="Cleaned Data",
                    index=False
                )

                pd.DataFrame(
                    st.session_state.cleaning_log
                ).to_excel(
                    writer,
                    sheet_name="Cleaning Records",
                    index=False
                )

            with download_col2:

                st.download_button(
                    label="⬇️ Download cleaned Excel",
                    data=excel_buffer.getvalue(),
                    file_name="cleaned_dataset.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                    use_container_width=True,
                    key="download_cleaned_excel"
                )

            # ---------------------------------------------
            # RESET CLEANING
            # ---------------------------------------------

            st.markdown("---")

            if st.button(
                "↩️ Reset cleaning results",
                use_container_width=True
            ):

                st.session_state.cleaned_df = None
                st.session_state.cleaning_log = []

                st.rerun()

            st.caption(
                "The original uploaded dataset is never modified. "
                "Cleaning operations are applied to a separate copy."
            )

    # -----------------------------------------------------
    # TAB 2: DATA QUALITY DETAILS
    # -----------------------------------------------------

    with tab2:

        st.markdown("### 🔍 Detailed quality findings")

        if (
            "detect_quality_issues" in observations
            and not show_observation_error(
                observations["detect_quality_issues"]
            )
        ):

            quality = observations[
                "detect_quality_issues"
            ]

            missing_values = quality.get(
                "missing_values",
                {}
            )

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

                st.success(
                    "No missing values were detected."
                )

            st.markdown(
                "#### Duplicate and constant columns"
            )

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
                "Data quality details are not available "
                "for this investigation."
            )

        st.markdown("---")

        st.markdown(
            "#### Potential outlier details"
        )

        if (
            "detect_anomalies" in observations
            and not show_observation_error(
                observations["detect_anomalies"]
            )
        ):

            anomaly = observations[
                "detect_anomalies"
            ].get("columns", {})

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

                st.info(
                    "No eligible outlier results are available."
                )

        else:

            st.info(
                "Outlier results are not available."
            )

        st.markdown("---")

        st.markdown(
            "#### 🧹 Non-destructive cleaning recommendations"
        )

        if (
            "create_cleaning_preview" in observations
            and not show_observation_error(
                observations["create_cleaning_preview"]
            )
        ):

            with st.expander(
                "View cleaning recommendations",
                expanded=True
            ):

                st.json(
                    observations["create_cleaning_preview"]
                )

            st.caption(
                "These are recommendations only. "
                "Your original dataset is not modified automatically."
            )

        else:

            st.info(
                "No cleaning preview is available."
            )

    # -----------------------------------------------------
    # TAB 3: EXECUTION TRACE AND LANGGRAPH
    # -----------------------------------------------------

    with tab3:

        st.markdown("### ⚙️ Agent execution trace")

        st.write(
            "This section shows the steps taken by the LangGraph "
            "agent during the investigation."
        )

        trace = result.get(
            "trace",
            []
        )

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

            st.info(
                "No execution trace is available."
            )

        st.markdown("---")

        st.markdown(
            "### 🔗 LangGraph workflow architecture"
        )

        st.write(
            "The workflow is organised into connected nodes. "
            "Each node performs a specific part of the investigation."
        )

        try:

            agent = create_agent()

            if hasattr(agent, "get_graph_mermaid"):

                mermaid_code = agent.get_graph_mermaid()

                if mermaid_code:

                    st.code(
                        mermaid_code,
                        language="mermaid"
                    )

                else:

                    st.info(
                        "The workflow diagram is not available."
                    )

            else:

                st.info(
                    "Add the get_graph_mermaid() method to agent.py "
                    "to display the workflow diagram here."
                )

        except Exception as e:

            st.warning(
                f"Could not load the LangGraph diagram: {e}"
            )

        st.markdown("---")

        st.markdown("### Agent investigation stages")

        st.markdown("""
        <div class="info-card">
            <p><b>1. Planning</b> — Qwen3 plans the investigation.</p>
            <p><b>2. Tool execution</b> — Python tools analyse the dataset.</p>
            <p><b>3. Observation</b> — The agent collects tool results.</p>
            <p><b>4. Explanation</b> — Qwen3 generates evidence-based findings.</p>
            <p><b>5. Memory</b> — Investigation information is stored in SQLite.</p>
        </div>
        """, unsafe_allow_html=True)

    # -----------------------------------------------------
    # TAB 4: FOLLOW-UP QUESTIONS
    # -----------------------------------------------------

    with tab4:

        st.markdown(
            "### 💬 Ask Qwen3 a follow-up question"
        )

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

                    st.warning(
                        "Please enter a question first."
                    )

                else:

                    try:

                        with st.spinner(
                            "Qwen3 is preparing an answer..."
                        ):

                            agent = create_agent()

                            answer = agent.answer_question(
                                question,
                                result
                            )

                            st.session_state.latest_answer = answer

                    except Exception as e:

                        st.error(
                            f"Unable to answer: {e}"
                        )

        if st.session_state.latest_answer is not None:

            st.markdown("#### Agent response")

            st.write(
                st.session_state.latest_answer
            )

    # -----------------------------------------------------
    # TAB 5: FEEDBACK AND SQLITE MEMORY
    # -----------------------------------------------------

    with tab5:

        st.markdown(
            "### ⭐ Help the agent improve"
        )

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

                    agent = create_agent()

                    agent.record_feedback(
                        rating,
                        comment,
                        category
                    )

                    st.session_state.last_feedback = {
                        "rating": rating,
                        "category": category,
                        "comment": comment
                    }

                    st.success(
                        "Feedback successfully stored in SQLite memory!"
                    )

                except Exception as e:

                    st.error(
                        f"Could not store feedback: {e}"
                    )

        if st.session_state.last_feedback is not None:

            st.caption(
                "Most recently submitted feedback"
            )

            st.write(
                f"**Rating:** "
                f"{st.session_state.last_feedback['rating']}/5"
            )

            st.write(
                f"**Category:** "
                f"{st.session_state.last_feedback['category']}"
            )

        st.markdown("---")

        st.markdown(
            "#### 🧠 How the memory works"
        )

        st.markdown("""
        <div class="info-card">
            <h4>SQLite Episodic Memory</h4>
            <p>
                The agent stores investigation information and
                user feedback in a local SQLite database.
                Feedback summaries can guide the planning
                of future investigations.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            "#### Feedback guidance"
        )

        try:

            agent = create_agent()

            if hasattr(agent, "feedback_guidance"):

                guidance = agent.feedback_guidance()

                st.write(
                    guidance if guidance else
                    "No feedback guidance is available yet."
                )

        except Exception as e:

            st.caption(
                f"Feedback guidance is currently unavailable: {e}"
            )


# =========================================================
# 11. EMPTY STATE
# =========================================================

if df is None:

    st.markdown("---")

    st.markdown(
        '<div class="section-label">Getting started</div>',
        unsafe_allow_html=True
    )

    st.subheader(
        "Your intelligent data assistant is ready."
    )

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
# 12. FOOTER
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
