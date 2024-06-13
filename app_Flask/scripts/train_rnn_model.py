import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer

import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Keras version:", tf.keras.__version__)

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

# Load the datasetS
df_modi = pd.read_csv("app_Flask/datasets/ModiRelatedTweetsWithSentiment.csv")
df_rahul = pd.read_csv("app_Flask/datasets/RahulRelatedTweetsWithSentiment.csv")
df_modi = df_modi.drop(df_modi.columns[0], axis=1)
df_rahul = df_rahul.drop(df_rahul.columns[0], axis=1)

# Apply text preprocessing to each tweet
df_modi['Cleaned Tweet'] = df_modi['Tweet'].fillna('').apply(preprocess_text)
df_rahul['Cleaned Tweet'] = df_rahul['Tweet'].fillna('').apply(preprocess_text)

# Encode sentiment labels (positive: 1, negative: 0)
df_modi['Encoded Emotion'] = df_modi['Emotion'].map({'pos': 1, 'neg': 0})
df_rahul['Encoded Emotion'] = df_rahul['Emotion'].map({'pos': 1, 'neg': 0})

# Add a column to indicate the source of the tweet
df_modi['Source'] = 'Modi'
df_rahul['Source'] = 'Rahul'
df_combined = pd.concat([df_modi, df_rahul], ignore_index=True)
df_combined.dropna(inplace=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_combined['Cleaned Tweet'], df_combined['Encoded Emotion'], test_size=0.2, random_state=42)

# Load the GloVe word embeddings
embeddings_index = {}
embedding_dim = 100

print("Loading GloVe embeddings...")

with open('app_Flask/datasets/glove.twitter.27B.100d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print("GloVe embeddings loaded.")

# Tokenize the text data
# Define maximum number of words and maximum word index
MAX_WORDS = 280
MAX_WORD_INDEX = 50000

# Initialize Tokenizer
tokenizer = Tokenizer()

# Fit Tokenizer on training data
tokenizer.fit_on_texts(X_train)

# Convert text sequences to sequences of integers
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

# Pad sequences to ensure uniform length
X_train_padded = pad_sequences(X_train_sequences, maxlen=MAX_WORDS)

print("Tokenization completed.")

# Create an embedding matrix
embedding_matrix = np.zeros((MAX_WORD_INDEX+1, embedding_dim))
for word, i in tokenizer.word_index.items():
    if i < MAX_WORD_INDEX:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in the embedding index will be all zeros
            embedding_matrix[i] = embedding_vector

print("Embedding matrix created.")
print(embedding_matrix.shape)

# Build the LSTM model
model = Sequential()
model.add(Embedding(input_dim=MAX_WORD_INDEX+1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=MAX_WORDS, trainable=True))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
print("Model built.")
import numpy as np

# Assuming X_train and y_train are your data variables
# Check the type of X_train and y_train
print(type(X_train_padded))
print(type(y_train))

# If they are not already NumPy arrays, convert them
if not isinstance(X_train_padded, np.ndarray):
    X_train = np.array(X_train)
if not isinstance(y_train, np.ndarray):
    y_train = np.array(y_train)

# Now, check their types again to confirm they are NumPy arrays
print(type(X_train_padded))
print(type(y_train))
# Assuming model is your Keras model containing an Embedding layer
embedding_layer = model.layers[0]  # Assuming the Embedding layer is the first layer
input_shape = embedding_layer.input_dim
print("Input shape of the Embedding layer:", input_shape)

# Train the model
epochs = 5
batch_size = 128

model.fit(X_train_padded, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=2)

print("Model trained.")

# Save the model
model.save('../models/sentiment_model.h5')
print("Model saved to disk")