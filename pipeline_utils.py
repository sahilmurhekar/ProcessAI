def generate_pipeline_code(processing_steps):
    code = """import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA

# Load your dataset
df = pd.read_csv('your_dataset.csv')

"""

    if 'delete_columns' in processing_steps:
        cols = processing_steps['delete_columns']
        code += f"# Delete columns\ndf.drop(columns={cols}, inplace=True)\n\n"

    if 'remove_duplicates' in processing_steps:
        code += "# Remove duplicates\ndf.drop_duplicates(inplace=True)\n\n"

    if 'null_handling' in processing_steps:
        for col, strat in processing_steps['null_handling'].items():
            strategy = strat['strategy']
            if strategy == 'mean':
                code += f"df['{col}'].fillna(df['{col}'].mean(), inplace=True)\n"
            elif strategy == 'median':
                code += f"df['{col}'].fillna(df['{col}'].median(), inplace=True)\n"
            elif strategy == 'most_frequent':
                code += f"df['{col}'].fillna(df['{col}'].mode()[0], inplace=True)\n"
            elif strategy == 'constant':
                fill = strat.get('fill_value', '0')
                code += f"df['{col}'].fillna({fill}, inplace=True)\n"
            elif strategy == 'drop':
                code += f"df.dropna(subset=['{col}'], inplace=True)\n"
        code += "\n"

    if 'standardize' in processing_steps:
        cols = processing_steps['standardize']
        code += f"scaler = StandardScaler()\ndf[{cols}] = scaler.fit_transform(df[{cols}])\n\n"

    if 'normalize' in processing_steps:
        cols = processing_steps['normalize']
        code += f"scaler = MinMaxScaler()\ndf[{cols}] = scaler.fit_transform(df[{cols}])\n\n"

    if 'log_transform' in processing_steps:
        for col in processing_steps['log_transform']:
            code += f"df['{col}'] = np.log1p(df['{col}'])\n"
        code += "\n"

    if 'encode_categorical' in processing_steps:
        enc = processing_steps['encode_categorical']
        cols = enc['columns']
        method = enc['method']
        if method == 'label':
            for col in cols:
                code += f"df['{col}'] = LabelEncoder().fit_transform(df['{col}'])\n"
        else:
            for col in cols:
                code += f"df = pd.get_dummies(df, columns=['{col}'], prefix='{col}')\n"
        code += "\n"

    if 'remove_outliers' in processing_steps:
        details = processing_steps['remove_outliers']
        for col in details['columns']:
            if details['method'] == 'zscore':
                code += f"df = df[np.abs((df['{col}'] - df['{col}'].mean()) / df['{col}'].std()) < {details['threshold']}]\n"
            elif details['method'] == 'iqr':
                code += f"""Q1 = df['{col}'].quantile(0.25)
Q3 = df['{col}'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['{col}'] >= Q1 - {details['threshold']} * IQR) & (df['{col}'] <= Q3 + {details['threshold']} * IQR)]\n"""
        code += "\n"

    if 'text_preprocess' in processing_steps:
        import_text_processing = """import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
"""
        code = import_text_processing + code
        options = processing_steps['text_preprocess']['options']
        for col in processing_steps['text_preprocess']['columns']:
            code += f"""
def clean_text(text):
    if isinstance(text, str):
"""
            if 'Lowercase' in options:
                code += "        text = text.lower()\n"
            if 'Remove Punctuation' in options:
                code += "        text = re.sub(r'[^\\w\\s]', '', text)\n"
            if 'Remove Stopwords' in options:
                code += "        text = ' '.join([w for w in text.split() if w not in stopwords.words('english')])\n"
            if 'Stemming' in options:
                code += "        text = ' '.join([PorterStemmer().stem(w) for w in text.split()])\n"
            if 'Lemmatization' in options:
                code += "        text = ' '.join([WordNetLemmatizer().lemmatize(w) for w in text.split()])\n"
            code += "    return text\n"
            code += f"df['{col}'] = df['{col}'].apply(clean_text)\n"

    if 'pca' in processing_steps:
        cols = processing_steps['pca']['columns']
        n = processing_steps['pca']['n_components']
        code += f"pca = PCA(n_components={n})\npca_result = pca.fit_transform(df[{cols}])\n"
        for i in range(n):
            code += f"df['pca_{i+1}'] = pca_result[:, {i}]\n"
        # Optional: remove original columns
        code += f"# df.drop(columns={cols}, inplace=True)\n"

    code += "\n# Final processed DataFrame: df"
    return code
