
import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Crop Recommendation System", page_icon="🌱")

st.title("🌾 Crop Recommendation System")
st.write("Enter soil and climate details to get the best crop recommendation for your land.")

# ---------------------- Data & Model Utilities ----------------------
FEATURE_ORDER = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
LABEL_COL = "label"

@st.cache_data(show_spinner=False)
def read_repo_csv(path: str):
    return pd.read_csv(path)

def validate_columns(df: pd.DataFrame):
    missing = [c for c in FEATURE_ORDER + [LABEL_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}. "
                         f"Expected columns: {FEATURE_ORDER + [LABEL_COL]}")
    # Ensure numeric dtypes for features
    for c in FEATURE_ORDER:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(subset=FEATURE_ORDER + [LABEL_COL], inplace=True)
    return df

@st.cache_resource(show_spinner=True)
def load_or_train_model(repo_csv_path: str, uploaded_csv: pd.DataFrame | None):
    # 1) If pickle exists, load it for fastest startup
    if os.path.exists("crop_model.pkl"):
        with open("crop_model.pkl", "rb") as f:
            return pickle.load(f), None

    # 2) If user uploaded a CSV this session, prefer that
    if uploaded_csv is not None:
        df = validate_columns(uploaded_csv.copy())
    # 3) Else try repo CSV
    elif repo_csv_path and os.path.exists(repo_csv_path):
        df = validate_columns(read_repo_csv(repo_csv_path).copy())
    else:
        return None, "No model or dataset found. Upload a CSV with columns: " + ", ".join(FEATURE_ORDER + [LABEL_COL])

    X = df[FEATURE_ORDER]
    y = df[LABEL_COL]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique())>1 else None)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Save for future runs
    with open("crop_model.pkl", "wb") as f:
        pickle.dump(model, f)

    st.info(f"Model trained on {len(df)} rows. Validation accuracy: {acc:.3f}")
    return model, None

# ---------------------- Sidebar: Data Options ----------------------
st.sidebar.header("📦 Data & Model")
uploaded = st.sidebar.file_uploader("Upload dataset CSV (optional)", type=["csv"], help="Must include columns: " + ", ".join(FEATURE_ORDER + [LABEL_COL]))
uploaded_df = None
if uploaded is not None:
    try:
        uploaded_df = pd.read_csv(uploaded)
        st.sidebar.success("Uploaded CSV loaded.")
    except Exception as e:
        st.sidebar.error(f"Failed to read CSV: {e}")

# Path to repo CSV if present
REPO_CSV = "crop_recommendation.csv"

model, err = load_or_train_model(REPO_CSV, uploaded_df)
if err:
    st.warning(err)

# ---------------------- Prediction UI ----------------------
col1, col2, col3 = st.columns(3)
with col1:
    N = st.number_input("Nitrogen (N)", min_value=0, max_value=300, value=50, step=1)
    P = st.number_input("Phosphorus (P)", min_value=0, max_value=300, value=50, step=1)
    K = st.number_input("Potassium (K)", min_value=0, max_value=300, value=50, step=1)
with col2:
    temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=25.0, step=0.1)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.1)
with col3:
    ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=1000.0, value=100.0, step=0.1)

can_predict = model is not None

if st.button("🔍 Recommend Crop", disabled=not can_predict):
    feats = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    try:
        pred = model.predict(feats)[0]
        st.success(f"🌱 Recommended Crop: **{pred}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.caption("Tip: If you see a warning about missing data/model, upload your dataset CSV from the sidebar or include a pre-trained `crop_model.pkl` in the repo.")
