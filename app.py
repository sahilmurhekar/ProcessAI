import streamlit as st
import pandas as pd
import plotly.express as px
import time
from data_preprocessor import DataPreprocessor, AIAnalyzer

st.set_page_config(page_title="AI Data Preprocessor", layout="wide")

st.title("AI-Powered Data Preprocessor")

# Sidebar
with st.sidebar:
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        if 'df' not in st.session_state:
            st.session_state.df = pd.read_csv(uploaded_file)
        st.write(f"Rows: {st.session_state.df.shape[0]}")
        st.write(f"Columns: {st.session_state.df.shape[1]}")
        if st.button("Generate AI Suggestions"):
            analyzer = AIAnalyzer(st.session_state.df)
            st.session_state.ai_suggestions = analyzer.analyze()
        if st.button("Process Data"):
            st.session_state.process = True
        if 'processed_df' in st.session_state:
            st.download_button("Download Processed CSV", st.session_state.processed_df.to_csv(index=False), "processed_data.csv")
            if st.button("Export Pipeline Code"):
                from pipeline_utils import generate_pipeline_code  # Ensure this is imported properly
                code = generate_pipeline_code(st.session_state.processing_steps)
                st.code(code, language="python")

if uploaded_file is None:
    st.info("Upload a CSV to get started.")
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(st.session_state.df.head())

if 'ai_suggestions' in st.session_state:
    st.subheader("AI Suggestions")
    for suggestion in st.session_state.ai_suggestions:
        st.info(suggestion)

# Tabbed interface for processing steps
tabs = st.tabs(["Basic", "Statistical", "Advanced"])

# Basic Operations
with tabs[0]:
    delete_cols = st.multiselect("Columns to delete", st.session_state.df.columns)
    remove_duplicates = st.checkbox("Remove Duplicates")
    handle_nulls = st.checkbox("Handle Nulls")
    null_strategies = {}
    if handle_nulls:
        null_cols = st.multiselect("Columns with nulls", st.session_state.df.columns[st.session_state.df.isnull().any()])
        for col in null_cols:
            strategy = st.selectbox(f"Strategy for {col}", ["mean", "median", "most_frequent", "constant", "drop"], key=col)
            fill = None
            if strategy == "constant":
                fill = st.text_input(f"Constant value for {col}", key=f"fill_{col}")
            null_strategies[col] = {"strategy": strategy, "fill_value": fill}

# Statistical Operations
with tabs[1]:
    num_cols = st.session_state.df.select_dtypes(include=['number']).columns
    standardize_cols = st.multiselect("Standardize (Z-score)", num_cols)
    normalize_cols = st.multiselect("Normalize (Min-Max)", num_cols)
    log_transform_cols = st.multiselect("Log Transform", num_cols)
    remove_outliers = st.checkbox("Remove Outliers")
    outlier_cols, outlier_method, outlier_threshold = [], "zscore", 3.0
    if remove_outliers:
        outlier_cols = st.multiselect("Columns for Outlier Removal", num_cols)
        outlier_method = st.radio("Method", ["zscore", "iqr"])
        outlier_threshold = st.slider("Threshold", 1.0, 5.0, 3.0)

# Advanced Operations
with tabs[2]:
    cat_cols = st.session_state.df.select_dtypes(include=['object']).columns
    encode_cols = st.multiselect("Encode Categorical", cat_cols)
    encode_method = st.radio("Encoding Method", ["one-hot", "label"])
    text_cols = st.multiselect("Preprocess Text Columns", cat_cols)
    text_options = st.multiselect("Text Processing Options", ["Lowercase", "Remove Punctuation", "Remove Stopwords", "Stemming", "Lemmatization"])
    pca_cols = st.multiselect("PCA Columns", num_cols)
    n_components = st.slider("Number of PCA Components", 1, len(pca_cols)) if pca_cols else 2

# Processing
if st.session_state.get("process"):
    st.session_state.process = False
    df = st.session_state.df.copy()
    preprocessor = DataPreprocessor(df=df)
    steps = {}
    
    if delete_cols:
        preprocessor.delete_columns(delete_cols)
        steps['delete_columns'] = delete_cols
    if remove_duplicates:
        preprocessor.remove_duplicates()
        steps['remove_duplicates'] = True
    if handle_nulls:
        for col, strat in null_strategies.items():
            preprocessor.handle_null_values([col], strat["strategy"], strat.get("fill_value"))
        steps['null_handling'] = null_strategies
    if standardize_cols:
        preprocessor.standardize_data(standardize_cols)
        steps['standardize'] = standardize_cols
    if normalize_cols:
        preprocessor.normalize_data(normalize_cols)
        steps['normalize'] = normalize_cols
    if log_transform_cols:
        preprocessor.log_transform(log_transform_cols)
        steps['log_transform'] = log_transform_cols
    if remove_outliers and outlier_cols:
        preprocessor.remove_outliers(outlier_cols, outlier_method, outlier_threshold)
        steps['remove_outliers'] = {"columns": outlier_cols, "method": outlier_method, "threshold": outlier_threshold}
    if encode_cols:
        preprocessor.encode_categorical(encode_cols, encode_method)
        steps['encode_categorical'] = {"columns": encode_cols, "method": encode_method}
    if text_cols:
        preprocessor.preprocess_text(text_cols, text_options)
        steps['text_preprocess'] = {"columns": text_cols, "options": text_options}
    if pca_cols:
        preprocessor.apply_pca(pca_cols, n_components)
        steps['pca'] = {"columns": pca_cols, "n_components": n_components}

    st.session_state.processed_df = preprocessor.get_processed_data()
    st.session_state.processing_steps = steps
    st.success("✅ Processing complete!")
    st.dataframe(st.session_state.processed_df.head())