import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Crop Recommendation System", page_icon="🌱", layout="centered")

st.title("🌱 Crop Recommendation System")

# File uploader
uploaded_file = st.file_uploader("Upload dataset CSV (optional)", type=["csv"])

# Default dataset from repo
DEFAULT_DATASET = "crop_recommendation.csv"

# Load dataset
try:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Using uploaded dataset.")
    else:
        df = pd.read_csv(DEFAULT_DATASET)
        st.info("📂 Using default dataset from repo.")
except Exception as e:
    st.error(f"❌ Could not load dataset: {e}")
    st.stop()

# Show dataset preview
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# Ensure dataset has rows
if df.shape[0] == 0:
    st.error("❌ Dataset is empty. Please upload a valid CSV.")
    st.stop()

# Split features and labels
try:
    X = df.drop(columns=['label'])
    y = df['label']
except KeyError:
    st.error("❌ 'label' column not found in dataset. Make sure your CSV has this column.")
    st.stop()

# Train-test split (safe)
if len(df) > 5:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
else:
    st.error("❌ Not enough data to train the model. Please upload a larger dataset.")
    st.stop()

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.success(f"✅ Model trained successfully! Accuracy: {acc:.2f}")

# Input section
st.subheader("🌾 Enter soil & climate details")

col1, col2, col3 = st.columns(3)

with col1:
    N = st.number_input("Nitrogen", min_value=0, max_value=200, value=50)
    P = st.number_input("Phosphorus", min_value=0, max_value=200, value=50)
    K = st.number_input("Potassium", min_value=0, max_value=200, value=50)

with col2:
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)

with col3:
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=6.5)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=100.0)

# Prediction
if st.button("🌿 Recommend Crop"):
    input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                              columns=X.columns)
    prediction = model.predict(input_data)[0]
    st.success(f"🌟 Recommended Crop: **{prediction}**")
