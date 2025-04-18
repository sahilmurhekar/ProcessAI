import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from data_preprocessor import DataPreprocessor, AIAnalyzer
from pipeline_utils import generate_pipeline_code

st.set_page_config(page_title="AI Data Preprocessor", layout="wide")

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .section-header {
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .stButton button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<div class='main-header'>🤖 AI-Powered Data Preprocessor</div>", unsafe_allow_html=True)

with st.sidebar:
    st.header("📁 Data Controls")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        if 'df' not in st.session_state:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.session_state.current_df = st.session_state.df.copy()
        
        if st.button("🔄 Reset to Original Data", key="reset_button"):
            st.session_state.current_df = st.session_state.df.copy()
            if 'processed_df' in st.session_state:
                del st.session_state.processed_df
            if 'processing_steps' in st.session_state:
                del st.session_state.processing_steps
            if 'ai_suggestions' in st.session_state:
                del st.session_state.ai_suggestions
            if 'pipeline_code' in st.session_state:
                del st.session_state.pipeline_code
            st.success("Reset to original data.")
        
        if st.button("🤖 Generate AI Suggestions", key="ai_suggestions_button"):
            analyzer = AIAnalyzer(st.session_state.current_df)
            st.session_state.ai_suggestions = analyzer.analyze()
        
        if st.button("⚙️ Process Data", key="process_data_button"):
            display_df = st.session_state.current_df
            preprocessor = DataPreprocessor(df=display_df.copy())
            steps = {}
            
            if 'delete_cols' in st.session_state and st.session_state.delete_cols:
                preprocessor.delete_columns(st.session_state.delete_cols)
                steps['delete_columns'] = st.session_state.delete_cols
            
            if 'remove_duplicates' in st.session_state and st.session_state.remove_duplicates:
                preprocessor.remove_duplicates()
                steps['remove_duplicates'] = True
            
            if 'null_strategies' in st.session_state and st.session_state.null_strategies:
                steps['null_handling'] = st.session_state.null_strategies
                for col, strat in st.session_state.null_strategies.items():
                    preprocessor.handle_null_values([col], strat["strategy"], strat.get("fill_value"))
            
            if 'standardize_cols' in st.session_state and st.session_state.standardize_cols:
                preprocessor.standardize_data(st.session_state.standardize_cols)
                steps['standardize'] = st.session_state.standardize_cols
            
            if 'normalize_cols' in st.session_state and st.session_state.normalize_cols:
                preprocessor.normalize_data(st.session_state.normalize_cols)
                steps['normalize'] = st.session_state.normalize_cols
            
            if 'log_transform_cols' in st.session_state and st.session_state.log_transform_cols:
                preprocessor.log_transform(st.session_state.log_transform_cols)
                steps['log_transform'] = st.session_state.log_transform_cols
            
            if 'outlier_info' in st.session_state and st.session_state.outlier_info['cols']:
                preprocessor.remove_outliers(
                    st.session_state.outlier_info['cols'], 
                    st.session_state.outlier_info['method'], 
                    st.session_state.outlier_info['threshold']
                )
                steps['remove_outliers'] = {
                    "columns": st.session_state.outlier_info['cols'],
                    "method": st.session_state.outlier_info['method'],
                    "threshold": st.session_state.outlier_info['threshold']
                }
            
            if 'encode_info' in st.session_state and st.session_state.encode_info['cols']:
                preprocessor.encode_categorical(
                    st.session_state.encode_info['cols'], 
                    st.session_state.encode_info['method']
                )
                steps['encode_categorical'] = {
                    "columns": st.session_state.encode_info['cols'], 
                    "method": st.session_state.encode_info['method']
                }
            
            if 'text_info' in st.session_state and st.session_state.text_info['cols']:
                preprocessor.preprocess_text(
                    st.session_state.text_info['cols'], 
                    st.session_state.text_info['options']
                )
                steps['text_preprocess'] = {
                    "columns": st.session_state.text_info['cols'], 
                    "options": st.session_state.text_info['options']
                }
            
            if 'pca_info' in st.session_state and st.session_state.pca_info['cols']:
                preprocessor.apply_pca(
                    st.session_state.pca_info['cols'], 
                    st.session_state.pca_info['n_components']
                )
                steps['pca'] = {
                    "columns": st.session_state.pca_info['cols'], 
                    "n_components": st.session_state.pca_info['n_components']
                }
            
            st.session_state.current_df = preprocessor.df
            st.session_state.processing_steps = steps
            st.session_state.pipeline_code = generate_pipeline_code(steps)
            st.success("✅ Data processed successfully!")
        
        if 'current_df' in st.session_state and st.session_state.current_df is not None:
            csv = st.session_state.current_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Processed Data", 
                          csv, 
                          file_name="processed_data.csv", 
                          mime="text/csv")
        else:
            st.button("📥 Download Processed Data", disabled=True)

if uploaded_file is not None:
    display_df = st.session_state.current_df
    
    main_tabs = st.tabs([
        "📊 Data Overview", 
        "⚙️ Processing Options", 
        "📈 Visualizations", 
        "🧠 AI Insights",
        "📤 Export"
    ])
    
    with main_tabs[0]:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Dataset Preview")
            st.dataframe(display_df.head(), use_container_width=True)
        
        with col2:
            st.subheader("Dataset Info")
            st.info(f"""
            **Rows:** {display_df.shape[0]}  
            **Columns:** {display_df.shape[1]}  
            **Memory Usage:** ouvau{display_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
            """)
            
            with st.expander("Show Dataset Summary", expanded=False):
                st.write("**Data Types:**")
                st.dataframe(pd.DataFrame(
                    {'Data Type': display_df.dtypes.value_counts().index.astype(str),
                     'Count': display_df.dtypes.value_counts().values}
                ))
                
                missing_data = display_df.isnull().sum()
                st.write("**Missing Values:**")
                if missing_data.sum() > 0:
                    missing_df = pd.DataFrame({
                        'Column': missing_data[missing_data > 0].index,
                        'Missing Count': missing_data[missing_data > 0].values,
                        'Missing %': (missing_data[missing_data > 0] / len(display_df) * 100).round(2)
                    })
                    st.dataframe(missing_df)
                else:
                    st.write("No missing values found!")
        
        st.subheader("Categorical Data Explorer")
        cat_cols = display_df.select_dtypes(include='object').columns
        if len(cat_cols) > 0:
            cat_col = st.selectbox("Select categorical column", cat_cols, key="cat_col_overview")
            if cat_col:
                value_counts = display_df[cat_col].value_counts()
                value_counts_df = pd.DataFrame({
                    "Value": value_counts.index,
                    "Count": value_counts.values,
                    "Percentage": (value_counts.values / len(display_df) * 100).round(2)
                })
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.dataframe(value_counts_df, use_container_width=True)
                with col2:
                    if len(value_counts) <= 15:
                        fig = px.pie(
                            value_counts_df, 
                            values='Count', 
                            names='Value', 
                            title=f"Distribution of {cat_col}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No categorical columns found.")
    
    with main_tabs[1]:
        proc_tabs = st.tabs(["Basic", "Statistical", "Advanced"])
        
        with proc_tabs[0]:
            with st.expander("🗑️ Column Management", expanded=False):
                st.session_state.delete_cols = st.multiselect(
                    "Select columns to delete", 
                    display_df.columns,
                    key="delete_cols_select"
                )
                if st.button("Preview Column Deletion", key="preview_col_delete"):
                    if st.session_state.delete_cols:
                        preview_df = display_df.drop(columns=st.session_state.delete_cols)
                        st.dataframe(preview_df.head())
                    else:
                        st.info("No columns selected for deletion.")
            
            with st.expander("🧹 Duplicate Handling", expanded=False):
                st.session_state.remove_duplicates = st.checkbox(
                    "Remove duplicate rows", 
                    key="remove_dups_check"
                )
                if st.session_state.remove_duplicates:
                    st.info(f"Found {display_df.duplicated().sum()} duplicate rows that will be removed.")
            
            with st.expander("🧩 Missing Value Treatment", expanded=False):
                null_cols = display_df.columns[display_df.isnull().any()]
                if len(null_cols) > 0:
                    st.session_state.handle_nulls = True
                    selected_null_cols = st.multiselect(
                        "Select columns with nulls to handle", 
                        null_cols,
                        key="null_cols_select"
                    )
                    
                    st.session_state.null_strategies = {}
                    for col in selected_null_cols:
                        st.write(f"**Handling strategy for: {col}**")
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            strategy = st.selectbox(
                                "Strategy", 
                                ["mean", "median", "most_frequent", "constant", "drop"],
                                key=f"strategy_{col}"
                            )
                        with col2:
                            fill = None
                            if strategy == "constant":
                                fill = st.text_input("Fill value", key=f"fill_{col}")
                            
                            missing_pct = (display_df[col].isnull().sum() / len(display_df) * 100).round(2)
                            st.info(f"Missing: {missing_pct}%")
                        
                        st.session_state.null_strategies[col] = {"strategy": strategy, "fill_value": fill}
                else:
                    st.success("No missing values detected in the dataset.")
                    st.session_state.handle_nulls = False
        
        with proc_tabs[1]:
            num_cols = display_df.select_dtypes(include=np.number).columns
            
            with st.expander("📊 Data Scaling", expanded=False):
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown("**Standardization (Z-score)**")
                    st.session_state.standardize_cols = st.multiselect(
                        "Select columns to standardize", 
                        num_cols,
                        key="std_cols"
                    )
                
                with col2:
                    st.markdown("**Normalization (Min-Max)**")
                    st.session_state.normalize_cols = st.multiselect(
                        "Select columns to normalize", 
                        num_cols,
                        key="norm_cols"
                    )
            
            with st.expander("🔢 Transform Data", expanded=False):
                st.session_state.log_transform_cols = st.multiselect(
                    "Select columns for Log Transform", 
                    num_cols,
                    key="log_cols"
                )
                
                if st.session_state.log_transform_cols:
                    preview_col = st.session_state.log_transform_cols[0]
                    if not display_df[preview_col].min() <= 0:
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            fig = px.histogram(
                                display_df, 
                                x=preview_col,
                                title=f"Original Distribution: {preview_col}"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            log_preview = display_df.copy()
                            log_preview[preview_col] = np.log(log_preview[preview_col])
                            fig = px.histogram(
                                log_preview, 
                                x=preview_col,
                                title=f"Log-Transformed: {preview_col}"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"Column {preview_col} contains zero or negative values - log transform requires positive values.")
            
            with st.expander("🚫 Outlier Treatment", expanded=False):
                st.session_state.remove_outliers = st.checkbox(
                    "Remove outliers", 
                    key="remove_outliers_check"
                )
                
                if st.session_state.remove_outliers:
                    outlier_cols = st.multiselect(
                        "Select columns for outlier removal", 
                        num_cols,
                        key="outlier_cols"
                    )
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        outlier_method = st.radio(
                            "Outlier detection method", 
                            ["zscore", "iqr"],
                            key="outlier_method"
                        )
                    
                    with col2:
                        if outlier_method == "zscore":
                            threshold_text = "Z-score threshold"
                            default_val = 3.0
                        else:
                            threshold_text = "IQR multiplier"
                            default_val = 1.5
                            
                        outlier_threshold = st.slider(
                            threshold_text, 
                            1.0, 5.0, default_val,
                            key="outlier_threshold"
                        )
                    
                    st.session_state.outlier_info = {
                        "cols": outlier_cols,
                        "method": outlier_method,
                        "threshold": outlier_threshold
                    }
                    
                    if outlier_cols:
                        first_col = outlier_cols[0]
                        data_series = display_df[first_col].dropna()
                        
                        if outlier_method == "zscore":
                            z_scores = np.abs((data_series - data_series.mean()) / data_series.std())
                            outliers = data_series[z_scores > outlier_threshold]
                        else:
                            q1 = data_series.quantile(0.25)
                            q3 = data_series.quantile(0.75)
                            iqr = q3 - q1
                            lower_bound = q1 - (outlier_threshold * iqr)
                            upper_bound = q3 + (outlier_threshold * iqr)
                            outliers = data_series[(data_series < lower_bound) | (data_series > upper_bound)]
                        
                        st.info(f"Will remove {len(outliers)} outliers ({len(outliers)/len(data_series)*100:.2f}%) from column {first_col}.")
                        
                        fig = px.box(display_df, y=first_col, title=f"Box Plot of {first_col} showing outliers")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.session_state.outlier_info = {"cols": [], "method": "zscore", "threshold": 3.0}
        
        with proc_tabs[2]:
            with st.expander("🔤 Categorical Encoding", expanded=False):
                encode_cols = st.multiselect(
                    "Select categorical columns to encode", 
                    cat_cols,
                    key="encode_cols"
                )
                
                encode_method = st.radio(
                    "Encoding Method", 
                    ["one-hot", "label"],
                    key="encode_method"
                )
                
                st.session_state.encode_info = {
                    "cols": encode_cols,
                    "method": encode_method
                }
                
                if encode_cols:
                    preview_col = encode_cols[0]
                    st.write(f"**Preview of encoding for column: {preview_col}**")
                    unique_values = display_df[preview_col].unique()
                    
                    if encode_method == "one-hot":
                        example_df = pd.get_dummies(display_df[preview_col]).head(5)
                        st.dataframe(example_df)
                    else:
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        values = display_df[preview_col].dropna().values
                        if len(values) > 0:
                            le.fit(values)
                            example = pd.DataFrame({
                                "Original": le.classes_,
                                "Encoded": le.transform(le.classes_)
                            })
                            st.dataframe(example)
            
            with st.expander("📝 Text Preprocessing", expanded=False):
                text_cols = st.multiselect(
                    "Select text columns to process", 
                    cat_cols,
                    key="text_cols"
                )
                
                text_options = st.multiselect(
                    "Text Processing Options", 
                    ["Lowercase", "Remove Punctuation", "Remove Stopwords", "Stemming", "Lemmatization"],
                    key="text_options"
                )
                
                st.session_state.text_info = {
                    "cols": text_cols,
                    "options": text_options
                }
                
                if text_cols and text_options:
                    preview_col = text_cols[0]
                    sample_text = display_df[preview_col].dropna().iloc[0] if len(display_df[preview_col].dropna()) > 0 else ""
                    
                    if sample_text:
                        st.write("**Text Processing Preview**")
                        processed_text = sample_text
                        
                        import string
                        import re
                        
                        if "Lowercase" in text_options:
                            processed_text = processed_text.lower()
                        
                        if "Remove Punctuation" in text_options:
                            processed_text = re.sub(f'[{string.punctuation}]', ' ', processed_text)
                        
                        if "Remove Stopwords" in text_options:
                            st.info("Stopwords would be removed in actual processing")
                        
                        if "Stemming" in text_options:
                            st.info("Words would be stemmed in actual processing")
                        
                        if "Lemmatization" in text_options:
                            st.info("Words would be lemmatized in actual processing")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Original Text:**")
                            st.text(sample_text)
                        
                        with col2:
                            st.markdown("**Processed Text:**")
                            st.text(processed_text)
            
            with st.expander("🔬 Dimensionality Reduction (PCA)", expanded=False):
                pca_cols = st.multiselect(
                    "Select numerical columns for PCA", 
                    num_cols,
                    key="pca_cols"
                )
                
                if pca_cols:
                    n_components = st.slider(
                        "Number of PCA Components", 
                        1, min(len(pca_cols), 10), 
                        min(2, len(pca_cols)),
                        key="n_components"
                    )
                    
                    st.session_state.pca_info = {
                        "cols": pca_cols,
                        "n_components": n_components
                    }
                    
                    if len(pca_cols) >= 2:
                        st.info(f"PCA will reduce {len(pca_cols)} features to {n_components} components.")
                else:
                    st.session_state.pca_info = {"cols": [], "n_components": 2}
    
    with main_tabs[2]:
        viz_type = st.radio(
            "Choose visualization type",
            ["Histogram", "Scatter Plot", "Box Plot", "Correlation Matrix", "Pair Plot"],
            horizontal=True
        )
        
        if viz_type == "Histogram":
            col1, col2 = st.columns([3, 1])
            with col1:
                hist_col = st.selectbox("Select column", num_cols, key="hist_col_viz")
                if hist_col:
                    fig = px.histogram(
                        display_df, 
                        x=hist_col, 
                        title=f"Histogram of {hist_col}",
                        marginal="box"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if hist_col:
                    st.write("**Statistics**")
                    stats = display_df[hist_col].describe()
                    st.dataframe(stats)
        
        elif viz_type == "Scatter Plot":
            if len(num_cols) >= 2:
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    x_col = st.selectbox("X-axis", num_cols, key="x_col_viz")
                with col2:
                    y_col = st.selectbox("Y-axis", num_cols, key="y_col_viz", index=min(1, len(num_cols)-1))
                with col3:
                    color_col = st.selectbox(
                        "Color by (optional)", 
                        ["None"] + list(display_df.columns),
                        key="color_col_viz"
                    )
                
                if x_col and y_col:
                    color = None if color_col == "None" else color_col
                    fig = px.scatter(
                        display_df, 
                        x=x_col, 
                        y=y_col, 
                        color=color,
                        title=f"{x_col} vs {y_col}",
                        trendline="ols" if color is None else None
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    corr = display_df[[x_col, y_col]].corr().iloc[0, 1]
                    st.info(f"Correlation coefficient: {corr:.4f}")
        
        elif viz_type == "Box Plot":
            col1, col2 = st.columns([3, 1])
            with col1:
                box_col = st.selectbox("Select numerical column", num_cols, key="box_col_viz")
                group_col = st.selectbox(
                    "Group by (optional)", 
                    ["None"] + list(cat_cols),
                    key="group_col_viz"
                )
                
                if box_col:
                    if group_col != "None" and group_col in display_df.columns:
                        top_categories = display_df[group_col].value_counts().nlargest(10).index
                        filtered_df = display_df[display_df[group_col].isin(top_categories)]
                        
                        fig = px.box(
                            filtered_df, 
                            x=group_col, 
                            y=box_col, 
                            title=f"Box Plot of {box_col} by {group_col}"
                        )
                    else:
                        fig = px.box(
                            display_df, 
                            y=box_col, 
                            title=f"Box Plot of {box_col}"
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if box_col:
                    st.write("**Statistics**")
                    stats = display_df[box_col].describe()
                    st.dataframe(stats)
        
        elif viz_type == "Correlation Matrix":
            if len(num_cols) >= 2:
                corr_cols = st.multiselect(
                    "Select columns for correlation matrix", 
                    num_cols,
                    default=list(num_cols)[:min(5, len(num_cols))],
                    key="corr_cols"
                )
                
                if corr_cols and len(corr_cols) >= 2:
                    corr_matrix = display_df[corr_cols].corr()
                    
                    fig = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        color_continuous_scale='RdBu_r',
                        title="Correlation Matrix"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    corr_pairs = []
                    for i in range(len(corr_cols)):
                        for j in range(i+1, len(corr_cols)):
                            corr_pairs.append({
                                'Variables': f"{corr_cols[i]} — {corr_cols[j]}",
                                'Correlation': corr_matrix.iloc[i, j]
                            })
                    
                    if corr_pairs:
                        corr_df = pd.DataFrame(corr_pairs)
                        corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
                        st.write("**Strongest Correlations:**")
                        st.dataframe(corr_df)
        
        elif viz_type == "Pair Plot":
            if len(num_cols) >= 2:
                pair_cols = st.multiselect(
                    "Select columns for pair plot (2-5 recommended)", 
                    num_cols,
                    default=list(num_cols)[:min(3, len(num_cols))],
                    key="pair_cols"
                )
                
                color_col = st.selectbox(
                    "Color by (optional)", 
                    ["None"] + list(cat_cols),
                    key="pair_color_col"
                )
                
                if pair_cols and len(pair_cols) >= 2:
                    if len(pair_cols) > 5:
                        st.warning("Selected too many columns. Limiting to first 5 for better performance.")
                        pair_cols = pair_cols[:5]
                    
                    color = None if color_col == "None" else color_col
                    if color and display_df[color].nunique() > 10:
                        top_categories = display_df[color].value_counts().nlargest(10).index
                        plot_df = display_df[display_df[color].isin(top_categories)]
                        st.info(f"Limiting to top 10 categories of {color} for better visibility.")
                    else:
                        plot_df = display_df
                    
                    fig = px.scatter_matrix(
                        plot_df,
                        dimensions=pair_cols,
                        color=color,
                        title="Pair Plot Matrix"
                    )
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
    
    with main_tabs[3]:
        if 'ai_suggestions' in st.session_state and st.session_state.ai_suggestions:
            st.success("AI analysis completed! Here are actionable insights for your dataset:")
            
            for i, suggestion in enumerate(st.session_state.ai_suggestions):
                with st.expander(f"Suggestion {i+1}", expanded=i==0):
                    st.info(suggestion)
        else:
            st.info("Click on 'Generate AI Suggestions' in the sidebar to analyze your data and get AI-driven recommendations.")
            
            with st.expander("What insights can the AI provide?", expanded=True):
                st.write("""
                The AI analyzer can help you with:
                
                - **Data quality issues**: Identifying missing values, outliers, and inconsistencies
                - **Feature recommendations**: Suggesting which features might be important
                - **Processing suggestions**: Recommending transformations that might improve your data
                - **Distribution analysis**: Detecting skewed distributions that might need transformation
                - **Correlation insights**: Finding important relationships between variables
                """)
            
            if st.button("🤖 Generate AI Suggestions", key="generate_ai_in_tab"):
                with st.spinner("Analyzing your data..."):
                    analyzer = AIAnalyzer(display_df)
                    st.session_state.ai_suggestions = analyzer.analyze()
    
    with main_tabs[4]:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📂 Processed Data")
            if 'processing_steps' in st.session_state:
                st.write("**Applied Processing Steps:**")
                
                steps_list = []
                if 'delete_columns' in st.session_state.processing_steps:
                    steps_list.append(f"- Deleted columns: {', '.join(st.session_state.processing_steps['delete_columns'])}")
                
                if 'remove_duplicates' in st.session_state.processing_steps:
                    steps_list.append(f"- Removed duplicate rows")
                
                if 'null_handling' in st.session_state.processing_steps:
                    null_strategies = st.session_state.processing_steps['null_handling']
                    steps_list.append(f"- Handled missing values in {len(null_strategies)} columns")
                
                if 'standardize' in st.session_state.processing_steps:
                    steps_list.append(f"- Standardized columns: {', '.join(st.session_state.processing_steps['standardize'])}")
                
                if 'normalize' in st.session_state.processing_steps:
                    steps_list.append(f"- Normalized columns: {', '.join(st.session_state.processing_steps['normalize'])}")
                
                if 'log_transform' in st.session_state.processing_steps:
                    steps_list.append(f"- Log-transformed columns: {', '.join(st.session_state.processing_steps['log_transform'])}")
                
                if 'remove_outliers' in st.session_state.processing_steps:
                    outlier_info = st.session_state.processing_steps['remove_outliers']
                    steps_list.append(f"- Removed outliers from columns: {', '.join(outlier_info['columns'])}")
                
                if 'encode_categorical' in st.session_state.processing_steps:
                    encode_info = st.session_state.processing_steps['encode_categorical']
                    steps_list.append(f"- {encode_info['method']} encoded columns: {', '.join(encode_info['columns'])}")
                
                if 'text_preprocess' in st.session_state.processing_steps:
                    text_info = st.session_state.processing_steps['text_preprocess']
                    steps_list.append(f"- Processed text in columns: {', '.join(text_info['columns'])}")
                
                if 'pca' in st.session_state.processing_steps:
                    pca_info = st.session_state.processing_steps['pca']
                    steps_list.append(f"- Applied PCA to columns: {', '.join(pca_info['columns'])} → {pca_info['n_components']} components")
                
                for step in steps_list:
                    st.write(step)
                
                st.write("**Processed Data Preview:**")
                st.dataframe(st.session_state.current_df.head(10), use_container_width=True)
                
                csv = st.session_state.current_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Processed CSV", 
                    csv, 
                    file_name="processed_data.csv", 
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("No processing has been applied yet. Use the Processing Options tab to prepare your data.")
        
        with col2:
            st.subheader("📝 Generated Python Code")
            if 'pipeline_code' in st.session_state:
                with st.expander("View Pipeline Code", expanded=True):
                    st.code(st.session_state.pipeline_code, language='python')
                
                st.download_button(
                    "💾 Download Python Script",
                    st.session_state.pipeline_code,
                    file_name="data_preprocessing_pipeline.py",
                    mime="text/plain",
                    use_container_width=True
                )
                
                st.info("""
                This code can be used to:
                - Reproduce your preprocessing pipeline
                - Apply the same transformations to new data
                - Integrate into your ML workflow
                """)
            else:
                st.info("Process your data first to generate the Python code for your preprocessing pipeline.")

else:
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h1>🤖 Welcome to AI-Powered Data Preprocessor</h1>
        <p style="font-size: 1.2rem; margin: 2rem 0;">
            Upload a CSV file in the sidebar to start preprocessing your data with AI assistance.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📊 Data Exploration
        - Preview and analyze your data
        - View statistics and distributions
        - Identify missing values and outliers
        """)
    
    with col2:
        st.markdown("""
        ### ⚙️ Data Processing
        - Clean and transform your data
        - Handle missing values
        - Encode categorical features
        - Scale numerical features
        """)
    
    with col3:
        st.markdown("""
        ### 🧠 AI Assistance
        - Get AI-powered suggestions
        - Automatically detect data issues
        - Generate Python code for your pipeline
        """)
    
    with st.expander("How to use this app", expanded=True):
        st.markdown("""
        1. **Upload Data**: Start by uploading your CSV file in the sidebar
        2. **Explore**: View your data and understand its properties
        3. **Process**: Select the transformations you want to apply
        4. **Generate AI Suggestions**: Get AI-powered recommendations
        5. **Apply**: Process your data with the selected transformations
        6. **Export**: Download your processed data and generated code
        """)