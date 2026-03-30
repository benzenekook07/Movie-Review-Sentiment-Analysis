import pandas as pd
import numpy as np
import re
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# download stopwords
nltk.download('stopwords')

# Load dataset
df = pd.read_csv('data/imdb.csv')

# Initialize
ps = PorterStemmer()

# Preprocessing function
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# Apply preprocessing
df['review'] = df['review'].apply(clean_text)

# Features & Labels
X = df['review']
y = df['sentiment']

# Convert labels
y = y.map({'positive': 1, 'negative': 0})

# Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
joblib.dump(model, 'models/model.pkl')
joblib.dump(tfidf, 'models/vectorizer.pkl')
