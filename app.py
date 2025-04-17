import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from data_preprocessor import DataPreprocessor, AIAnalyzer

st.set_page_config(page_title="AI Data Preprocessor", layout="wide")
st.title("🤖 AI-Powered Data Preprocessor")

st.header("📁 Upload Your Data")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    if 'df' not in st.session_state:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.session_state.current_df = st.session_state.df.copy()

    if st.button("🔄 Reset to Original Data"):
        st.session_state.current_df = st.session_state.df.copy()
        st.session_state.pop('processed_df', None)
        st.session_state.pop('processing_steps', None)
        st.session_state.pop('ai_suggestions', None)
        st.success("Reset to original data.")
        st.rerun()

    display_df = st.session_state.current_df
    st.subheader("📊 Dataset Preview")
    st.dataframe(display_df.head())

    st.subheader("📐 Dataset Info")
    st.write(f"**Rows:** {display_df.shape[0]}")
    st.write(f"**Columns:** {display_df.shape[1]}")

    st.subheader("🧮 Categorical Column Value Counts")
    cat_cols = display_df.select_dtypes(include='object').columns
    if len(cat_cols) > 0:
        cat_col = st.selectbox("Select categorical column", cat_cols)
        if cat_col:
            value_counts = display_df[cat_col].value_counts()
            value_counts_df = pd.DataFrame({
                "Value": value_counts.index,
                "Count": value_counts.values,
                "Percentage": (value_counts.values / len(display_df) * 100).round(2)
            })
            st.dataframe(value_counts_df)
    else:
        st.write("No categorical columns found.")

    if 'ai_suggestions' in st.session_state:
        st.subheader("💡 AI Suggestions")
        for suggestion in st.session_state.ai_suggestions:
            st.info(suggestion)

    tabs = st.tabs(["⚙️ Basic", "📈 Statistical", "🧠 Advanced", "📤 Export"])

    with tabs[0]:
        delete_cols = st.multiselect("🗑️ Columns to delete", display_df.columns)
        remove_duplicates = st.checkbox("🧹 Remove Duplicates")
        handle_nulls = st.checkbox("🧹 Handle Null Values")
        null_strategies = {}
        if handle_nulls:
            null_cols = display_df.columns[display_df.isnull().any()]
            selected_null_cols = st.multiselect("Select columns with nulls", null_cols)
            for col in selected_null_cols:
                strategy = st.selectbox(f"Strategy for {col}", ["mean", "median", "most_frequent", "constant", "drop"], key=col)
                fill = st.text_input(f"Constant value for {col}", key=f"fill_{col}") if strategy == "constant" else None
                null_strategies[col] = {"strategy": strategy, "fill_value": fill}

    with tabs[1]:
        num_cols = display_df.select_dtypes(include=np.number).columns
        standardize_cols = st.multiselect("📊 Standardize (Z-score)", num_cols)
        normalize_cols = st.multiselect("📉 Normalize (Min-Max)", num_cols)
        log_transform_cols = st.multiselect("🔢 Log Transform", num_cols)
        remove_outliers = st.checkbox("🚫 Remove Outliers")

        outlier_cols, outlier_method, outlier_threshold = [], "zscore", 3.0
        if remove_outliers:
            outlier_cols = st.multiselect("Columns for Outlier Removal", num_cols)
            outlier_method = st.radio("Method", ["zscore", "iqr"])
            outlier_threshold = st.slider("Threshold", 1.0, 5.0, 3.0)

    with tabs[2]:
        encode_cols = st.multiselect("🔡 Encode Categorical Columns", cat_cols)
        encode_method = st.radio("Encoding Method", ["one-hot", "label"])
        text_cols = st.multiselect("📝 Preprocess Text Columns", cat_cols)
        text_options = st.multiselect("Text Processing Options", ["Lowercase", "Remove Punctuation", "Remove Stopwords", "Stemming", "Lemmatization"])
        pca_cols = st.multiselect("🔬 PCA Columns", num_cols)
        n_components = st.slider("Number of PCA Components", 1, len(pca_cols)) if pca_cols else 2

    with tabs[3]:
        st.subheader("📂 Extracted Python Code")
        if 'pipeline_code' in st.session_state:
            st.code(st.session_state.pipeline_code, language='python')
        else:
            st.info("No processing steps applied yet.")

    st.subheader("📊 Data Visualization")
    viz_tabs = st.tabs(["Histogram", "Scatter", "Box", "Summary"])

    with viz_tabs[0]:
        hist_col = st.selectbox("Column for Histogram", num_cols)
        if hist_col:
            fig = px.histogram(display_df, x=hist_col, title=f"Histogram of {hist_col}")
            st.plotly_chart(fig)

    with viz_tabs[1]:
        if len(num_cols) >= 2:
            x_col = st.selectbox("X-axis", num_cols)
            y_col = st.selectbox("Y-axis", num_cols)
            if x_col and y_col:
                fig = px.scatter(display_df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
                st.plotly_chart(fig)

    with viz_tabs[2]:
        box_col = st.selectbox("Column for Box Plot", num_cols)
        if box_col:
            fig = px.box(display_df, y=box_col, title=f"Box Plot of {box_col}")
            st.plotly_chart(fig)

    with viz_tabs[3]:
        stats_type = st.radio("Choose Summary Type", ["Numerical", "Categorical"])
        if stats_type == "Numerical":
            stats_cols = st.multiselect("Select numerical columns", num_cols)
            if stats_cols:
                summary_df = display_df[stats_cols].describe().T
                st.dataframe(summary_df)
        else:
            freq_cols = st.multiselect("Select categorical columns", cat_cols)
            for col in freq_cols:
                counts = display_df[col].value_counts()
                summary = pd.DataFrame({
                    'Value': counts.index,
                    'Frequency': counts.values,
                    'Percentage': (counts.values / len(display_df) * 100).round(2)
                })
                st.write(f"### {col}")
                st.dataframe(summary)

    st.subheader("🛠️ Processing Controls")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🤖 Generate AI Suggestions"):
            analyzer = AIAnalyzer(display_df)
            st.session_state.ai_suggestions = analyzer.analyze()
            st.rerun()

    with col2:
        if st.button("\u2705 Process Data"):
            preprocessor = DataPreprocessor(df=display_df.copy())
            steps = {}

            if delete_cols:
                preprocessor.delete_columns(delete_cols)
                steps['deleted'] = delete_cols
            if remove_duplicates:
                preprocessor.remove_duplicates()
                steps['deduplicated'] = True
            if handle_nulls:
                for col, strat in null_strategies.items():
                    preprocessor.handle_null_values([col], strat["strategy"], strat.get("fill_value"))
                steps['null_handled'] = null_strategies
            if standardize_cols:
                preprocessor.standardize_data(standardize_cols)
                steps['standardized'] = standardize_cols
            if normalize_cols:
                preprocessor.normalize_data(normalize_cols)
                steps['normalized'] = normalize_cols
            if log_transform_cols:
                preprocessor.log_transform(log_transform_cols)
                steps['log_transformed'] = log_transform_cols
            if remove_outliers and outlier_cols:
                preprocessor.remove_outliers(outlier_cols, outlier_method, outlier_threshold)
                steps['outliers_removed'] = {
                    "columns": outlier_cols,
                    "method": outlier_method,
                    "threshold": outlier_threshold
                }
            if encode_cols:
                preprocessor.encode_categorical(encode_cols, encode_method)
                steps['encoded'] = {"columns": encode_cols, "method": encode_method}
            if text_cols:
                preprocessor.preprocess_text(text_cols, text_options)
                steps['text_processed'] = {"columns": text_cols, "options": text_options}
            if pca_cols:
                preprocessor.apply_pca(pca_cols, n_components)
                steps['pca'] = {"columns": pca_cols, "n_components": n_components}

            st.session_state.current_df = preprocessor.df
            st.session_state.processing_steps = steps
            st.session_state.pipeline_code = preprocessor.get_pipeline_code()  # <-- must be implemented in your class
            st.success("✅ Data processed successfully!")
            st.rerun()

    with col3:
        if st.button("📅 Download Processed Data"):
            processed_df = st.session_state.get('current_df', None)
            if processed_df is not None:
                csv = processed_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download CSV", csv, file_name="processed_data.csv", mime="text/csv")
            else:
                st.warning("No processed data to download.")

    # 🚨 Show Processed Data Below Buttons
    if 'processing_steps' in st.session_state:
        st.subheader("✅ Processed Data Preview")
        st.dataframe(st.session_state.current_df)
