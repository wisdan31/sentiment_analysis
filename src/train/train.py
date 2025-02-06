import pandas as pd
import numpy as np
import re
import nltk
import logging
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import KeyedVectors

nltk.download('stopwords')
nltk.download("punkt")
nltk.download('punkt_tab')
nltk.download('wordnet')

STOPWORDS = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

def lemmatize_words(text):
    return [lemmatizer.lemmatize(word) for word in text]

def remove_stopwords(text):
    return [word for word in text if word not in STOPWORDS]

def tokenize(text):
    tokens = word_tokenize(text)
    return tokens

def clean_text(text):
    """
    Cleans text out of all redundant characters
    """
    text = re.sub(r'\W', ' ', str(text))  # Remove special characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)  # Remove single characters
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)  # Remove single characters from start
    text = re.sub(r'\s+', ' ', text, flags=re.I)  # Replace multiple spaces with single space
    text = re.sub(r'^b\s+', '', text)  # Remove prefixed 'b'
    text = text.lower()
    return text


def get_average_word2vec(tokens_list, model, num_features=300):
    # Filter tokens that exist in the Word2Vec vocabulary
    tokens = [word for word in tokens_list if word in model.key_to_index]
    
    if len(tokens) == 0:  # If no valid words in the model, return a zero vector
        return np.zeros(num_features)
    
    # Calculate the average of word vectors for the tokens
    word_vectors = [model[word] for word in tokens]
    avg_vector = np.mean(word_vectors, axis=0)
    
    return avg_vector

def process_data():
    """
    Processes data for further training
    """
    train_data_path = "src/data/raw/train.csv"
    test_data_path = "src/data/raw/test.csv"

    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    logging.info("Data loaded")

    X_train = train_data["review"]
    y_train = train_data["sentiment"]
    X_test = test_data["review"]
    y_test = test_data["sentiment"]

    # Clean text
    X_train = X_train.apply(clean_text)
    X_test = X_test.apply(clean_text)

    logging.info("Cleared text")

    # Tokenize text
    X_train = X_train.apply(tokenize)
    X_test = X_test.apply(tokenize)

    logging.info("Tokenized text")

    X_train = X_train.apply(remove_stopwords)
    X_test = X_test.apply(remove_stopwords)

    logging.info("Removed stopwords")

    X_train = X_train.apply(lemmatize_words)
    X_test = X_test.apply(lemmatize_words)

    logging.info("Lemmatized words")

    # Uncomment when deploying and comment other one(!)
    # word2vec = api.load("word2vec-google-news-300")
    # word2vec = KeyedVectors.load("notebooks/word2vec-google-news-300.kv", mmap='r')

    logging.debug(f"First 5 rows of X_train: {X_train.head()}")

    y_train = y_train.apply(lambda x: 1 if x == 'positive' else 0)
    y_test = y_test.apply(lambda x: 1 if x == 'positive' else 0)

    train_data = pd.concat([X_train, y_train], axis = 1)
    test_data = pd.concat([X_test, y_test], axis = 1)

    os.makedirs("src/data/processed", exist_ok=True)

    train_data.to_csv("src/data/processed/train.csv")
    test_data.to_csv("src/data/processed/test.csv")



