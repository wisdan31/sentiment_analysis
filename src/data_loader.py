import pandas as pd
import os
from urllib.request import urlretrieve

def download_data():
    """
    Downloads train and test datasets from cloud
    """
    data_dir = "src/data"
    train_filename = os.path.join(data_dir, "train.csv")
    test_filename = os.path.join(data_dir, "test.csv")
    train_data_path = "https://drive.google.com/uc?export=download&id=16CHSMU1ffqMZc__bTZngjZOjREghWKid"
    test_data_path = "https://drive.google.com/uc?export=download&id=1tUMuS0ol19IUO8zgOcHpHWLV2pQVA8qZ"

    os.makedirs(data_dir, exist_ok=True)

    urlretrieve(train_data_path, train_filename)
    urlretrieve(test_data_path, test_filename)

if __name__ == "__main__":
    download_data()