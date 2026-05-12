import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -------------------------------
# Load datasets
# -------------------------------

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine datasets
df = pd.concat([fake, true], ignore_index=True)

# Use text + label
X = df["text"]
y = df["label"]

# -------------------------------
# Convert text into vectors
# -------------------------------

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

X_vector = vectorizer.fit_transform(X)

# -------------------------------
# Split data
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_vector, y, test_size=0.2, random_state=42
)

# -------------------------------
# Train model
# -------------------------------

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# -------------------------------
# Streamlit UI
# -------------------------------

st.title("📰 Fake News Detection System")

st.write("Enter a news article to check if it is REAL or FAKE")

news = st.text_area("Enter News Content")

if st.button("Check News"):

    news_vector = vectorizer.transform([news])

    prediction = model.predict(news_vector)

    if prediction[0] == 1:
        st.success("✅ This News is REAL")
    else:
        st.error("❌ This News is FAKE")