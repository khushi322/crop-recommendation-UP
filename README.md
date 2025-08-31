
# 🌾 Crop Recommendation — Streamlit App

Deploy-ready Streamlit app that predicts the best crop based on soil and weather conditions.

## 📦 Project Structure
```
streamlit-crop-app/
├── app.py
├── requirements.txt
├── crop_recommendation.csv   # optional: your dataset (N,P,K,temperature,humidity,ph,rainfall,label)
└── crop_model.pkl            # optional: a pre-trained model
```

You can provide **either** a `crop_model.pkl` **or** a `crop_recommendation.csv`. If neither is present, you can upload a CSV at runtime via the sidebar.

### Expected CSV Columns
- N, P, K, temperature, humidity, ph, rainfall, label

## ▶️ Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy on Streamlit Cloud (step-by-step)
1. Create a new GitHub repo and push these files.
2. Add either `crop_model.pkl` **or** `crop_recommendation.csv` to the repo (or plan to upload a CSV from the app sidebar after deploy).
3. Go to https://share.streamlit.io/ and connect your repo.
4. Set the app entry point to `app.py` and deploy.
5. Open your app URL. If you didn't include a model/CSV, upload a CSV from the sidebar to train a model on the fly.
