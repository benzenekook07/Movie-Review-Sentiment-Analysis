import streamlit as st
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

# Load model
model = joblib.load('models/model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

ps = PorterStemmer()

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# UI
st.title("🎬 Movie Review Sentiment Analyzer")

review = st.text_area("Enter your movie review:")

if st.button("Analyze"):
    cleaned = clean_text(review)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)

    if prediction[0] == 1:
        st.success("Positive 😊")
    else:
        st.error("Negative 😠")
