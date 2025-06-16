import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor


def load_data():
    train_path = os.path.join("data", "processed", "train.csv")
    return pd.read_csv(train_path)


def build_pipeline(df):
    num_attribs = list(df.drop("median_house_value", axis=1).select_dtypes(include=["int64", "float64"]).columns)
    cat_attribs = list(df.drop("median_house_value", axis=1).select_dtypes(include=["object"]).columns)

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_attribs),
    ])

    return full_pipeline, num_attribs + cat_attribs


def main():
    df = load_data()
    pipeline, _ = build_pipeline(df)
    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"].copy()
    X_prepared = pipeline.fit_transform(X)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_prepared, y)
    os.makedirs("models", exist_ok=True)
    joblib.dump((pipeline, model), os.path.join("models", "model.joblib"))
    print("Model saved to models/model.joblib")


if __name__ == "__main__":
    main()
