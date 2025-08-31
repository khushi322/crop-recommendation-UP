import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# Helper: Train model
# -----------------------------
def train_model(df):
    X = df.drop('label', axis=1)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc


# -----------------------------
# Helper: Load or Train Model
# -----------------------------
def load_or_train_model(repo_csv_path: str, uploaded_df: pd.DataFrame = None):
    try:
        if uploaded_df is not None:
            df = uploaded_df
            st.sidebar.info("✅ Using uploaded dataset.")
        else:
            df = pd.read_csv(repo_csv_path)
            st.sidebar.info("📂 Using default repo dataset.")

        model, acc = train_model(df)
        return model, acc, None
    except Exception as e:
        return None, None, str(e)


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🌱 Crop Recommendation System")

# File uploader (optional)
uploaded = st.sidebar.file_uploader("Upload dataset CSV (optional)", type=["csv"])
uploaded_df = None
if uploaded is not None:
    try:
        uploaded_df = pd.read_csv(uploaded)
        st.sidebar.success("✅ Uploaded CSV loaded.")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to read CSV: {e}")

# Load dataset and train
REPO_CSV = "crop_recommendation.csv"
model, acc, err = load_or_train_model(REPO_CSV, uploaded_df)

if err:
    st.error(f"Error while training model: {err}")
    st.stop()

st.success(f"✅ Model trained successfully with accuracy: {acc:.2f}")

# -----------------------------
# Prediction Inputs
# -----------------------------
st.subheader("🔮 Predict Suitable Crop")

N = st.number_input("Nitrogen", 0, 200, 90)
P = st.number_input("Phosphorus", 0, 200, 42)
K = st.number_input("Potassium", 0, 200, 43)
temperature = st.number_input("Temperature (°C)", 0.0, 100.0, 20.0)
humidity = st.number_input("Humidity (%)", 0.0, 100.0, 80.0)
ph = st.number_input("pH", 0.0, 14.0, 6.5)
rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 200.0)

if st.button("Predict Crop"):
    try:
        features = [[N, P, K, temperature, humidity, ph, rainfall]]
        prediction = model.predict(features)
        st.success(f"🌾 Recommended Crop: **{prediction[0]}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
