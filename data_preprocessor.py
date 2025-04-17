import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import os
import logging
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('preprocessing.log'), logging.StreamHandler()]
)

class AIAnalyzer:
    def __init__(self, df):
        """Initialize with a DataFrame to analyze."""
        
        self.df = df
        self.numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        self.categorical_cols = df.select_dtypes(include=['object']).columns

    def analyze(self):
        """Analyze dataset and return preprocessing suggestions."""
        suggestions = []

        # Check for missing values
        missing = self.df.isnull().sum()
        missing_cols = missing[missing > 0]
        if not missing_cols.empty:
            for col, count in missing_cols.items():
                if col in self.numerical_cols:
                    suggestions.append(f"Impute missing values in '{col}' with mean or median (missing: {count})")
                elif col in self.categorical_cols:
                    suggestions.append(f"Impute missing values in '{col}' with most frequent or constant (missing: {count})")

        # Check for numerical features
        for col in self.numerical_cols:
            if self.df[col].std() > 2 * self.df[col].mean():
                suggestions.append(f"Standardize numerical feature '{col}' due to high variance")
            if (self.df[col] > 0).all() and self.df[col].skew() > 1:
                suggestions.append(f"Apply log transformation to '{col}' due to high skewness")
            if self.df[col].nunique() > 20:
                suggestions.append(f"Consider discretizing numerical column '{col}' into bins")
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))).sum()
            if outliers > 0:
                suggestions.append(f"Remove outliers in '{col}' (detected: {outliers})")

        # Check for categorical variables
        for col in self.categorical_cols:
            suggestions.append(f"Encode categorical variable '{col}' using one-hot or label encoding")
            if self.df[col].str.len().mean() > 50:
                suggestions.append(f"Preprocess text column '{col}' (e.g., lowercase, remove special characters)")

        # Check for class imbalance
        for col in self.df.columns:
            if self.df[col].dtype in ['int64', 'float64', 'object']:
                value_counts = self.df[col].value_counts()
                if len(value_counts) > 1 and value_counts.min() / value_counts.max() < 0.3:
                    suggestions.append(f"Resample dataset using '{col}' as target due to class imbalance")

        # Check for duplicates
        if self.df.duplicated().sum() > 0:
            suggestions.append(f"Remove {self.df.duplicated().sum()} duplicate rows")

        # Check for high dimensionality
        if len(self.numerical_cols) > 10:
            suggestions.append(f"Apply PCA to numerical columns {', '.join(self.numerical_cols)} to reduce dimensionality")

        return suggestions if suggestions else ["No specific preprocessing needed. Consider visualization."]

class DataPreprocessor:
    def __init__(self, file_path=None, df=None):
        """Initialize with CSV file path or DataFrame."""
        try:
            if df is not None:
                self.df = df.copy()
            elif file_path is not None:
                self.df = pd.read_csv(file_path)
            else:
                raise ValueError("Either a DataFrame or a file_path must be provided.")
            self.original_df = self.df.copy()
            self.numerical_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
            self.categorical_cols = self.df.select_dtypes(include=['object']).columns
            logging.info("Initialized DataPreprocessor")
        except Exception as e:
            logging.error(f"Failed to initialize: {e}")
            raise


    def get_pipeline_code(self):
        lines = ["import pandas as pd"]
        lines.append("df = pd.read_csv('your_file.csv')")

        if hasattr(self, 'steps'):
            for step, val in self.steps.items():
                if step == "deleted":
                    lines.append(f"df.drop(columns={val}, inplace=True)")
                elif step == "deduplicated":
                    lines.append("df.drop_duplicates(inplace=True)")
                elif step == "null_handled":
                    for col, strat in val.items():
                        if strat["strategy"] == "mean":
                            lines.append(f"df['{col}'].fillna(df['{col}'].mean(), inplace=True)")
                        elif strat["strategy"] == "median":
                            lines.append(f"df['{col}'].fillna(df['{col}'].median(), inplace=True)")
                        elif strat["strategy"] == "most_frequent":
                            lines.append(f"df['{col}'].fillna(df['{col}'].mode()[0], inplace=True)")
                        elif strat["strategy"] == "constant":
                            lines.append(f"df['{col}'].fillna({strat['fill_value']}, inplace=True)")
                        elif strat["strategy"] == "drop":
                            lines.append(f"df.dropna(subset=['{col}'], inplace=True)")
                # Add code generation for other steps similarly

        lines.append("df.to_csv('processed_output.csv', index=False)")
        return "\n".join(lines)


    def handle_null_values(self, columns, strategy='mean', fill_value=None):
        """Handle missing values in specified columns."""
        try:
            columns = [col for col in columns if col in self.df.columns]
            if not columns:
                raise ValueError("No valid columns for null handling")
            if strategy in ['mean', 'median', 'most_frequent']:
                imputer = SimpleImputer(strategy=strategy)
                self.df[columns] = pd.DataFrame(imputer.fit_transform(self.df[columns]), 
                                               columns=columns, index=self.df.index)
            elif strategy == 'constant':
                if fill_value is None:
                    raise ValueError("fill_value must be provided for constant strategy")
                self.df[columns] = self.df[columns].fillna(fill_value)
            elif strategy == 'drop':
                self.df.dropna(subset=columns, inplace=True)
            else:
                raise ValueError(f"Unsupported strategy: {strategy}")
            self._update_column_types()
            logging.info(f"Handled null values in columns {columns} with strategy: {strategy}")
        except Exception as e:
            logging.error(f"Error in handle_null_values: {e}")
            raise

    def standardize_data(self, columns):
        """Standardize specified numerical features."""
        try:
            columns = [col for col in columns if col in self.numerical_cols]
            if not columns:
                raise ValueError("No valid numerical columns for standardization")
            scaler = StandardScaler()
            self.df[columns] = pd.DataFrame(scaler.fit_transform(self.df[columns]), 
                                           columns=columns, index=self.df.index)
            logging.info(f"Standardized columns: {columns}")
        except Exception as e:
            logging.error(f"Error in standardize_data: {e}")
            raise

    def normalize_data(self, columns):
        """Normalize specified numerical features."""
        try:
            columns = [col for col in columns if col in self.numerical_cols]
            if not columns:
                raise ValueError("No valid numerical columns for normalization")
            scaler = MinMaxScaler()
            self.df[columns] = pd.DataFrame(scaler.fit_transform(self.df[columns]), 
                                           columns=columns, index=self.df.index)
            logging.info(f"Normalized columns: {columns}")
        except Exception as e:
            logging.error(f"Error in normalize_data: {e}")
            raise

    def log_transform(self, columns):
        """Apply log transformation to specified columns."""
        try:
            columns = [col for col in columns if col in self.numerical_cols]
            if not columns:
                raise ValueError("No valid numerical columns for log transformation")
            for col in columns:
                if (self.df[col] <= 0).any():
                    raise ValueError(f"Column {col} contains non-positive values, cannot apply log transformation")
                self.df[col] = np.log1p(self.df[col])
            logging.info(f"Applied log transformation to columns: {columns}")
        except Exception as e:
            logging.error(f"Error in log_transform: {e}")
            raise

    def encode_categorical(self, columns, method='one-hot'):
        """Encode specified categorical variables."""
        try:
            columns = [col for col in columns if col in self.categorical_cols]
            if not columns:
                raise ValueError("No valid categorical columns for encoding")
            if method == 'one-hot':
                self.df = pd.get_dummies(self.df, columns=columns, drop_first=True)
            elif method == 'label':
                for col in columns:
                    le = LabelEncoder()
                    self.df[col] = le.fit_transform(self.df[col].astype(str))
            else:
                raise ValueError(f"Unsupported encoding method: {method}")
            self._update_column_types()
            logging.info(f"Encoded categorical columns: {columns} using {method}")
        except Exception as e:
            logging.error(f"Error in encode_categorical: {e}")
            raise

    def delete_columns(self, columns):
        """Delete specified columns from the dataset."""
        try:
            columns = [col for col in columns if col in self.df.columns]
            if not columns:
                raise ValueError("No valid columns to delete")
            self.df.drop(columns=columns, inplace=True)
            self._update_column_types()
            logging.info(f"Deleted columns: {columns}")
        except Exception as e:
            logging.error(f"Error in delete_columns: {e}")
            raise

    def remove_duplicates(self):
        """Remove duplicate rows from the dataset."""
        try:
            initial_rows = len(self.df)
            self.df.drop_duplicates(inplace=True)
            removed_rows = initial_rows - len(self.df)
            logging.info(f"Removed {removed_rows} duplicate rows")
        except Exception as e:
            logging.error(f"Error in remove_duplicates: {e}")
            raise

    def discretize_data(self, columns, bins):
        """Discretize specified numerical columns into bins."""
        try:
            columns = [col for col in columns if col in self.numerical_cols]
            if not columns:
                raise ValueError("No valid numerical columns for discretization")
            for col in columns:
                self.df[col] = pd.cut(self.df[col], bins=bins, labels=False, include_lowest=True)
            self._update_column_types()
            logging.info(f"Discretized columns: {columns} into {bins} bins")
        except Exception as e:
            logging.error(f"Error in discretize_data: {e}")
            raise

    def preprocess_text(self, columns, options=[]):
        """Preprocess specified text columns with selected options."""
        try:
            columns = [col for col in columns if col in self.categorical_cols]
            if not columns:
                raise ValueError("No valid text columns for preprocessing")

            stop_words = set(stopwords.words('english'))
            stemmer = PorterStemmer()
            lemmatizer = WordNetLemmatizer()

            def clean_text(text):
                if isinstance(text, str):
                    if 'Lowercase' in options:
                        text = text.lower()
                    if 'Remove Punctuation' in options:
                        text = re.sub(r'[^a-zA-Z0-9\\s]', '', text)
                    if 'Remove Stopwords' in options:
                        text = ' '.join([w for w in text.split() if w not in stop_words])
                    if 'Stemming' in options:
                        text = ' '.join([stemmer.stem(w) for w in text.split()])
                    if 'Lemmatization' in options:
                        text = ' '.join([lemmatizer.lemmatize(w) for w in text.split()])
                return text

            for col in columns:
                self.df[col] = self.df[col].astype(str).apply(clean_text)

            logging.info(f"Preprocessed text columns: {columns} with options: {options}")
        except Exception as e:
            logging.error(f"Error in preprocess_text: {e}")
            raise

    def apply_pca(self, columns, n_components):
        """Apply PCA to specified numerical columns."""
        try:
            columns = [col for col in columns if col in self.numerical_cols]
            if not columns:
                raise ValueError("No valid numerical columns for PCA")
            if len(columns) < n_components:
                raise ValueError(f"Number of components ({n_components}) exceeds number of selected columns")
            pca = PCA(n_components=n_components)
            pca_data = pca.fit_transform(self.df[columns])
            pca_columns = [f'PC{i+1}' for i in range(n_components)]
            self.df = pd.concat([self.df.drop(columns=columns), 
                               pd.DataFrame(pca_data, columns=pca_columns, index=self.df.index)], axis=1)
            self._update_column_types()
            logging.info(f"Applied PCA to columns {columns} with {n_components} components")
        except Exception as e:
            logging.error(f"Error in apply_pca: {e}")
            raise

    def detect_outliers(self, columns, method='zscore', threshold=3):
        """Detect outliers in specified columns."""
        try:
            columns = [col for col in columns if col in self.numerical_cols]
            if not columns:
                raise ValueError("No valid numerical columns for outlier detection")
            outliers = np.zeros(self.df.shape[0], dtype=bool)
            if method == 'zscore':
                z_scores = np.abs(stats.zscore(self.df[columns]))
                outliers = (z_scores > threshold).any(axis=1)
            elif method == 'iqr':
                Q1 = self.df[columns].quantile(0.25)
                Q3 = self.df[columns].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((self.df[columns] < (Q1 - 1.5 * IQR)) | 
                           (self.df[columns] > (Q3 + 1.5 * IQR))).any(axis=1)
            else:
                raise ValueError(f"Unsupported outlier detection method: {method}")
            logging.info(f"Detected {outliers.sum()} outliers in columns {columns} using {method}")
            return outliers
        except Exception as e:
            logging.error(f"Error in detect_outliers: {e}")
            raise

    def remove_outliers(self, columns, method='zscore', threshold=3):
        """Remove outliers from specified columns."""
        try:
            outliers = self.detect_outliers(columns, method, threshold)
            self.df = self.df[~outliers]
            logging.info(f"Removed {outliers.sum()} outliers from columns {columns}")
        except Exception as e:
            logging.error(f"Error in remove_outliers: {e}")
            raise

    def resample_data(self, target_column, strategy='smote'):
        """Resample imbalanced dataset."""
        try:
            if target_column not in self.df.columns:
                raise ValueError(f"Target column {target_column} not found")
            X = self.df.drop(columns=[target_column])
            y = self.df[target_column]
            if strategy == 'smote':
                sampler = SMOTE(random_state=42)
            elif strategy == 'random_oversample':
                sampler = RandomOverSampler(random_state=42)
            elif strategy == 'random_undersample':
                sampler = RandomUnderSampler(random_state=42)
            else:
                raise ValueError(f"Unsupported resampling strategy: {strategy}")
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            self.df = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), 
                               pd.Series(y_resampled, name=target_column)], axis=1)
            self._update_column_types()
            logging.info(f"Resampled data using {strategy} for target: {target_column}")
        except Exception as e:
            logging.error(f"Error in resample_data: {e}")
            raise

    def visualize_data(self, output_dir='plots'):
        """Generate visualizations for data exploration."""
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            plots = []

            # Histogram for numerical columns
            for col in self.numerical_cols:
                plt.figure(figsize=(8, 6))
                sns.histplot(self.df[col], kde=True)
                plt.title(f'Distribution of {col}')
                plot_path = os.path.join(output_dir, f'histogram_{col}.png')
                plt.savefig(plot_path)
                plt.close()
                plots.append(plot_path)

            # Correlation heatmap
            if len(self.numerical_cols) > 1:
                plt.figure(figsize=(10, 8))
                sns.heatmap(self.df[self.numerical_cols].corr(), annot=True, cmap='coolwarm')
                plt.title('Correlation Heatmap')
                plot_path = os.path.join(output_dir, 'correlation_heatmap.png')
                plt.savefig(plot_path)
                plt.close()
                plots.append(plot_path)

            # Boxplot for numerical columns
            if len(self.numerical_cols) > 0:
                plt.figure(figsize=(10, 6))
                self.df[self.numerical_cols].boxplot()
                plt.title('Boxplot of Numerical Features')
                plt.xticks(rotation=45)
                plot_path = os.path.join(output_dir, 'boxplot_numerical.png')
                plt.savefig(plot_path)
                plt.close()
                plots.append(plot_path)

            # Bar plot for categorical columns
            for col in self.categorical_cols:
                plt.figure(figsize=(8, 6))
                self.df[col].value_counts().plot(kind='bar')
                plt.title(f'Value Counts of {col}')
                plt.xticks(rotation=45)
                plot_path = os.path.join(output_dir, f'barplot_{col}.png')
                plt.savefig(plot_path)
                plt.close()
                plots.append(plot_path)

            logging.info(f"Visualizations saved in {output_dir}")
            return plots
        except Exception as e:
            logging.error(f"Error in visualize_data: {e}")
            raise

    def save_processed_data(self, output_path):
        """Save the processed dataset to a CSV file."""
        try:
            self.df.to_csv(output_path, index=False)
            logging.info(f"Processed data saved to {output_path}")
        except Exception as e:
            logging.error(f"Error in save_processed_data: {e}")
            raise

    def get_processed_data(self):
        """Return the processed DataFrame."""
        return self.df

    def _update_column_types(self):
        """Update numerical and categorical column lists."""
        self.numerical_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns
        logging.info("Updated column types")