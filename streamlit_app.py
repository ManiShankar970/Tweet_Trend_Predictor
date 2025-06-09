# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
from textblob import TextBlob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load model and scaler
model = joblib.load('tweet_trend_model.pkl')
scaler = joblib.load('tweet_scaler.pkl')

st.title("🔮 Tweet Trend Predictor")
st.write("Enter the tweet's engagement details below to predict if it will trend.")

# Input fields
retweets = st.number_input("Number of Retweets", min_value=0, value=0)
likes = st.number_input("Number of Likes", min_value=0, value=0)
tweet_text = st.text_area("Tweet Text (for sentiment analysis)", "Type here...")
hour_posted = st.selectbox("Hour Tweet was Posted (24h)", list(range(24)))

# Validate input
if not tweet_text.strip():
    st.warning("Please enter tweet text for sentiment analysis.")
    st.stop()

# Process sentiment
sentiment = TextBlob(tweet_text).sentiment.polarity

# Prepare features
data = {
    'Retweets': retweets,
    'Likes': likes,
    'sentiment': sentiment,
    'hour_posted': hour_posted
}
data['engagement_score'] = data['Retweets'] * 0.6 + data['Likes'] * 0.4

data['peak_hour'] = 1 if data['hour_posted'] in [12, 13, 18, 19, 20] else 0

features = ['Retweets', 'Likes', 'sentiment', 'engagement_score', 'hour_posted', 'peak_hour']
X = pd.DataFrame([data])[features]
X_scaled = scaler.transform(X)

# Predict button
if st.button("Predict Trend"):
    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1]

    st.subheader("📊 Prediction Result")
    if pred:
        st.success(f"✅ This tweet is likely to trend! (Confidence: {prob:.2%})")
    else:
        st.warning(f"❌ This tweet may not trend. (Confidence: {prob:.2%})")

    # Show bar chart of feature values
    st.subheader("🔍 Feature Breakdown")
    chart_data = pd.DataFrame(data, index=[0])[features]
    st.bar_chart(chart_data.T)

    # Log prediction to CSV
    log_entry = pd.DataFrame({
        'timestamp': [datetime.now()],
        'retweets': [retweets],
        'likes': [likes],
        'sentiment': [sentiment],
        'hour_posted': [hour_posted],
        'prediction': [int(pred)],
        'confidence': [float(prob)]
    })

    log_file = "prediction_log.csv"
    if os.path.exists(log_file):
        log_entry.to_csv(log_file, mode='a', index=False, header=False)
    else:
        log_entry.to_csv(log_file, index=False)

    st.info("Prediction logged successfully ✅")
