import joblib
import re
from nltk.stem import WordNetLemmatizer
import pandas as pd

def lemmatize_words(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_ipc(x):
    if isinstance(x, list):
        return x
    if isinstance(x, (set, tuple)):
        return list(x)
    if pd.isna(x):
        return []
    return []


def preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the input DataFrame by handling missing values,
    encoding categorical variables, and vectorizing text data.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame with raw data.
    
    Returns:
    pd.DataFrame: Preprocessed DataFrame ready for modeling.
    """
    
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # ========== 1. ONE-HOT ENCODING ==========
    #ohe_columns = joblib.load("artefacts/ohe_columns.pkl")
    
    # CRITICAL FIX: Remove duplicates from saved OHE columns
    #ohe_columns_unique = list(dict.fromkeys(ohe_columns))
    
    ohe_cols = ['bail_type', 'crime_type', 'region', 'accused_gender', 'prior_cases']
    df_ohe = pd.get_dummies(df[ohe_cols], drop_first=True, prefix=ohe_cols)
    
    # Reindex with deduplicated columns
    #df_ohe = df_ohe.reindex(columns=ohe_columns_unique, fill_value=0)
    
    # Drop original categorical columns
    df = df.drop(columns=ohe_cols)
    df = pd.concat([df.reset_index(drop=True),
                    df_ohe.reset_index(drop=True)], axis=1)
    
    # ========== 2. BOOLEAN ENCODING ==========
    bool_cols = ['bail_cancellation_case', 'landmark_case',
                 'parity_argument_used', 'bias_flag']
    
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].replace({
                'True': 1, 'False': 0,
                'Yes': 1, 'No': 0,
                True: 1, False: 0
            }).fillna(0).astype(int)
    
    # ========== 3. IPC SECTIONS ENCODING ==========
    mlb = joblib.load('./artefacts/ipc_mlb.pkl')
    
    df['ipc_sections'] = df['ipc_sections'].apply(normalize_ipc)
    ipc_encoded = mlb.transform(df['ipc_sections'])
    
    # Create IPC DataFrame with unique column names
    ipc_columns = [c for c in mlb.classes_]
    ipc_df = pd.DataFrame(ipc_encoded, columns=ipc_columns, index=df.index)
    
    df = df.drop(columns=['ipc_sections'])
    df = pd.concat([df, ipc_df], axis=1)
    # ========== 4. TF-IDF VECTORIZATION ==========
    tfidf = joblib.load('./artefacts/tfidf_vectorizer.pkl')
    
    df['facts_clean'] = df['facts'].apply(clean_text)
    df['facts_clean'] = df['facts_clean'].apply(lemmatize_words)
    
    tfidf_matrix = tfidf.transform(df['facts_clean'])
    
    # CRITICAL FIX: Get feature names and handle potential duplicates properly
    tfidf_features = tfidf.get_feature_names_out()
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_features, index=df.index)
    # Drop original text columns
    df = df.drop(columns=['facts', 'facts_clean'], errors='ignore')
    df = pd.concat([df.reset_index(drop=True),
                    tfidf_df.reset_index(drop=True)], axis=1)
    print(f"Preprocessing complete. DataFrame shape: {df.shape}")
    return df


def predict_bail_score(df: pd.DataFrame, mlp) -> float:
    """
    Predict bail scores using the trained model.
    
    Parameters:
    df (pd.DataFrame): Preprocessed DataFrame
    mlp: Trained model object
    
    Returns:
    numpy.ndarray: Predicted bail scores
    """
    return mlp.predict(df)