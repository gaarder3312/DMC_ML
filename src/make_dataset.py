import os
import tarfile
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("data", "raw")
HOUSING_URL = os.path.join(DOWNLOAD_ROOT, "datasets/housing/housing.tgz")


def fetch_housing_data():
    os.makedirs(HOUSING_PATH, exist_ok=True)
    tgz_path = os.path.join(HOUSING_PATH, "housing.tgz")
    if not os.path.exists(tgz_path):
        print("Downloading dataset...")
        urllib.request.urlretrieve(HOUSING_URL, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=HOUSING_PATH)
    housing_tgz.close()


def load_housing_data():
    csv_path = os.path.join(HOUSING_PATH, "housing.csv")
    return pd.read_csv(csv_path)


def main():
    fetch_housing_data()
    housing = load_housing_data()
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    train_set.to_csv(os.path.join("data", "processed", "train.csv"), index=False)
    test_set.to_csv(os.path.join("data", "processed", "test.csv"), index=False)
    print("Data saved to data/processed")


if __name__ == "__main__":
    main()
