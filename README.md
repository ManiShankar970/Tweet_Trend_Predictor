# 🔮 Tweet Trend Predictor

A Machine Learning-powered web application that predicts whether a tweet is likely to trend based on engagement metrics and sentiment analysis. The application is built using **Python**, **Scikit-learn**, and **Streamlit**, providing an interactive interface for real-time tweet trend prediction.

---

## 📌 Project Overview

Social media platforms generate millions of tweets every day. Predicting whether a tweet will become popular helps marketers, businesses, and content creators optimize their posting strategy.

This project analyzes tweet engagement features such as:

- Number of Retweets
- Number of Likes
- Tweet Sentiment
- Posting Hour
- Engagement Score
- Peak Posting Time

and predicts whether the tweet is likely to trend.

#EXAMPLE PICS:
<img width="1572" height="911" alt="image" src="https://github.com/user-attachments/assets/f70f151c-96c8-4e43-9ba4-81762cc6a77b" />


<img width="1337" height="742" alt="image" src="https://github.com/user-attachments/assets/451db313-5b8c-4bb2-be55-079f58bdf4a8" />



---

## 🚀 Features

- 📊 Predicts tweet trending probability using a trained Machine Learning model.
- 😊 Performs automatic sentiment analysis using **TextBlob**.
- 📈 Calculates tweet engagement score.
- 🕒 Identifies whether the tweet was posted during peak engagement hours.
- 📉 Displays feature breakdown using an interactive bar chart.
- 💾 Automatically logs every prediction into a CSV file.
- 🌐 Simple and responsive Streamlit web interface.

---

## 🖥️ Application Preview

### Main Interface

- Enter tweet engagement details
- Provide tweet text
- Select posting hour
- Click **Predict Trend**

### Prediction Result

The application displays:

- Trend Prediction
- Prediction Confidence
- Feature Breakdown Chart
- Prediction Logging Status

---

## 📂 Project Structure

```
Tweet-Trend-Predictor/
│
├── streamlit_app.py          # Main Streamlit application
├── tweet_trend_model.pkl     # Trained Machine Learning model
├── tweet_scaler.pkl          # Feature scaler
├── prediction_log.csv        # Stores prediction history
├── requirements.txt
├── README.md
```

---

## ⚙️ Technologies Used

| Category | Technology |
|----------|------------|
| Language | Python |
| Web Framework | Streamlit |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Model Serialization | Joblib |
| NLP | TextBlob |
| Visualization | Matplotlib, Seaborn |
| Data Logging | CSV |

---

## 📊 Features Used for Prediction

The model predicts trends using the following input features:

| Feature | Description |
|----------|-------------|
| Retweets | Number of retweets |
| Likes | Number of likes |
| Sentiment | Sentiment polarity extracted from tweet text |
| Engagement Score | `(Retweets × 0.6) + (Likes × 0.4)` |
| Hour Posted | Hour when the tweet was posted |
| Peak Hour | Whether the tweet was posted during peak engagement time |

---

## 😊 Sentiment Analysis

The application uses **TextBlob** to calculate tweet sentiment.

Example:

Positive Tweet

```
I am very happy to see this amazing event!
```

Sentiment Score

```
1.0
```

Negative Tweet

```
This is the worst experience ever.
```

Sentiment Score

```
-1.0
```

Neutral Tweet

```
Today is Monday.
```

Sentiment Score

```
0.0
```

---

## 📈 Engagement Score Formula

```
Engagement Score =
(Retweets × 0.6) + (Likes × 0.4)
```

This score represents the overall engagement level of a tweet.

---

## 🕒 Peak Hour Detection

The application considers the following hours as peak engagement periods:

```
12 PM
1 PM
6 PM
7 PM
8 PM
```

If the tweet is posted during these hours:

```
Peak Hour = 1
```

Otherwise:

```
Peak Hour = 0
```

---

## 📊 Prediction Output

The application displays:

- ✅ Tweet Likely to Trend
or

- ❌ Tweet May Not Trend

along with

- Prediction Confidence
- Feature Breakdown Chart

Example:

```
Prediction

Tweet may not trend.

Confidence:
51%
```

---

## 📁 Prediction Logging

Every prediction is automatically stored inside

```
prediction_log.csv
```

Logged Information:

- Timestamp
- Retweets
- Likes
- Sentiment Score
- Posting Hour
- Prediction
- Confidence Score

Example

| Timestamp | Retweets | Likes | Prediction |
|-----------|----------|-------|------------|
|2026-07-09|9|30|Not Trend|

---

## 🛠️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Tweet-Trend-Predictor.git
```

Move into project folder

```bash
cd Tweet-Trend-Predictor
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run streamlit_app.py
```

---

## 📦 Requirements

```
streamlit
pandas
numpy
textblob
scikit-learn
joblib
matplotlib
seaborn
```

---

## 🎯 Future Improvements

- Twitter API integration
- Deep Learning based prediction
- BERT sentiment analysis
- Real-time tweet fetching
- Dashboard analytics
- User authentication
- Cloud deployment
- Multiple ML model comparison

---

## 📚 Learning Outcomes

Through this project, I learned:

- Machine Learning model deployment
- Feature Engineering
- Sentiment Analysis using NLP
- Streamlit Web Application Development
- Data Visualization
- Model Serialization with Joblib
- CSV Data Logging
- Building interactive ML applications

---


## ⭐ If you found this project useful, don't forget to Star this repository!
