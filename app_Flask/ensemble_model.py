import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import pickle
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore

# Load the VADER model
sid = SentimentIntensityAnalyzer()

# Load the trained RNN model
rnn_model = load_model('app_Flask/models/sentiment_model.h5')

nltk.download('stopwords')
nltk.download('wordnet')

# Text preprocessing function
def preprocess_text(text):
    # Check if text is a string
    if isinstance(text, str):
        # Convert text to lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove emojis (if needed)
        text = text.encode('ascii', 'ignore').decode('ascii')
        # Tokenization and removal of stopwords
        stop_words = set(stopwords.words('english'))
        words = [word for word in text.split() if word not in stop_words]
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        return " ".join(words)
    else:
        return ""  # Return empty string for non-string values

# Load the GloVe word embeddings

embeddings_index = {}
embedding_dim = 100

with open('app_Flask/datasets/glove.twitter.27B.100d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Tokenize the text data

MAX_WORDS = 280
MAX_WORD_INDEX = 50000
embedding_dim = 100

def tokenize_text(text):

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)

    tokenized_text = tokenizer.texts_to_sequences(text)
    padded_text = pad_sequences(tokenized_text, maxlen=MAX_WORDS)

    embedding_matrix = np.zeros((MAX_WORD_INDEX, embedding_dim))
    for word, i in tokenizer.word_index.items():
        if i < MAX_WORD_INDEX:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # Words not found in the embedding index will be all zeros
                embedding_matrix[i] = embedding_vector

    return padded_text

# Function to analyze sentiment using VADER
def analyze_sentiment_vader(text):
    ss = sid.polarity_scores(text)
    return ss['compound']


# Function to analyze sentiment using RNN
def analyze_sentiment_rnn(padded_text):
    pred = rnn_model.predict(padded_text)
    normalized_predictions = (pred - 0.5) * 2
    return normalized_predictions

# Function to combine predictions from VADER and RNN
def combine_predictions(text):
    rnn_weight = 0.4
    vader_weight = 1-rnn_weight
    text = preprocess_text(text)
    padded_text = tokenize_text([text])
    prediction_rnn = analyze_sentiment_rnn(padded_text)
    prediction_vader = analyze_sentiment_vader(text)
    prediction = (rnn_weight * prediction_rnn) + (vader_weight * prediction_vader)
    return prediction


# def predict_sentiment(text):
#     df = pd.read_csv('app_Flask/datasets/IndianElection19TwitterData.csv')

#     df_filtered = df.sample(frac=1)
#     df_filtered = df[df['Tweet'].str.contains(QUERY)]
#     df_filtered.drop(['Unnamed: 0'], axis=1, inplace=True)
#     df_filtered.reset_index(drop=True, inplace=True)
#     df_filtered.head()

#     pos_count = 0
#     neg_count = 0
#     for i in range(min(100, len(df_filtered))):
#         text = df_filtered['Tweet'][i]
        
#         prediction = combine_predictions(text)
#         if prediction > 0:
#             pos_count = pos_count + 1
#         elif prediction < 0:
#             neg_count = neg_count + 1

#     print('Positive tweets:', pos_count)
#     print('Negative tweets:', neg_count)

# QUERY = 'modi'
# predict_sentiment(QUERY)


