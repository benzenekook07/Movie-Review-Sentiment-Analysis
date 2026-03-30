
🎬 Movie Review Sentiment Analyzer
📌 Overview

This project is a simple Machine Learning application that analyzes movie reviews and predicts whether the sentiment is positive or negative.

It uses Natural Language Processing (NLP) techniques to process text data and classify it using a trained model.

🚀 Features
Predicts sentiment of movie reviews
Uses NLP for text preprocessing
Implements machine learning model (Naive Bayes)
Accepts custom user input
Simple and interactive interface using Streamlit
🧠 Tech Stack
Python
pandas, numpy
scikit-learn
nltk
streamlit
📂 Dataset

The dataset used is the IMDb Movie Reviews dataset containing labeled reviews.

⚙️ Project Workflow
Data collection
Data preprocessing (cleaning text, removing stopwords, stemming)
Feature extraction using TF-IDF
Model training using Naive Bayes
Model evaluation
Prediction on new data
🚀 How to Run the Project

Follow the steps below to run the project locally:

1. Clone the repository
git clone https://github.com/benzenekook07/movie-sentiment-analyzer.git
cd movie-sentiment-analyzer
2. Install dependencies
pip install -r requirements.txt
3. Train the model
python train_model.py
4. Run the application
streamlit run app.py
📊 Example

Input:

This movie was really amazing and enjoyable

Output:

Positive 😊
📁 Project Structure
movie-sentiment-analyzer/
│
├── data/
├── models/
├── notebooks/
│
├── train_model.py
├── app.py
├── requirements.txt
├── README.md
🎯 Future Improvements
Improve accuracy using deep learning models
Add better UI design
Deploy the project online
Add support for neutral sentiment
📌 Note

This project is created for learning purposes and demonstrates the use of NLP and Machine Learning in sentiment analysis.

