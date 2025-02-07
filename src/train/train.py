import pandas as pd
import numpy as np
import re
import nltk
import logging
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from gensim.models import KeyedVectors
from gensim import downloader as api
import joblib


nltk.download('stopwords')
nltk.download("punkt")
nltk.download('punkt_tab')
nltk.download('wordnet')

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

STOPWORDS = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(SRC_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(SRC_DIR, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "outputs", "models")

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
    train_data_path = os.path.join(RAW_DATA_DIR, "train.csv")
    test_data_path = os.path.join(RAW_DATA_DIR, "test.csv")

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
    word2vec = api.load("word2vec-google-news-300")
    # word2vec = KeyedVectors.load("notebooks/word2vec-google-news-300.kv", mmap='r')

    X_train = np.array([get_average_word2vec(review, word2vec) for review in X_train])
    X_test = np.array([get_average_word2vec(review, word2vec) for review in X_test])

    logging.info("Completed word embedding")


    y_train = y_train.apply(lambda x: 1 if x == 'positive' else 0)
    y_test = y_test.apply(lambda x: 1 if x == 'positive' else 0)

    # Assuming that X_train and X_test are numpy arrays
    X_train = pd.DataFrame(X_train)  # No need to specify columns, as it has 300 features
    X_test = pd.DataFrame(X_test)


    # Concatenate the sentiment labels with X_train and X_test
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    # Ensure y_train and y_test have the 'sentiment' column name
    train_data["sentiment"] = y_train
    test_data["sentiment"] = y_test


    logging.debug(f"First 5 rows of X_train: {train_data.head()}")


    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    train_data.to_csv(os.path.join(PROCESSED_DATA_DIR, "train.csv"))
    test_data.to_csv(os.path.join(PROCESSED_DATA_DIR, "test.csv"))

    logging.info("Saved data")


def train_model():
    reviews_data = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "train.csv"))
    X = reviews_data.iloc[:, :-1]  # Select all rows and all columns except the last one
    y = reviews_data.iloc[:, -1]   # Select the last column (sentiment labels)


    print(y)


    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)  

    train_accuracy = model.score(X, y)
    logging.info(f"Training Accuracy: {train_accuracy:.4f}")

    os.makedirs((MODELS_DIR), exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, "model_01.pkl"))
    logging.info("Model saved successfully.")

    



if __name__ == "__main__":
    process_data()
    train_model()