import os
import json
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def load_data():
    test_path = os.path.join("data", "processed", "test.csv")
    return pd.read_csv(test_path)


def load_model():
    return joblib.load(os.path.join("models", "model.joblib"))


def main():
    df = load_data()
    pipeline, model = load_model()
    X_test = df.drop("median_house_value", axis=1)
    y_test = df["median_house_value"].copy()
    X_prepared = pipeline.transform(X_test)
    predictions = model.predict(X_prepared)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    os.makedirs(os.path.join("data", "scores"), exist_ok=True)
    metrics = {"mse": mse, "r2": r2}
    with open(os.path.join("data", "scores", "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
