import argparse
import joblib
import pandas as pd


def load_model(path="models/model.joblib"):
    return joblib.load(path)


def parse_args():
    parser = argparse.ArgumentParser(description="Make predictions with trained model")
    parser.add_argument("--input", help="Path to CSV file with input data", required=True)
    parser.add_argument("--output", help="Path to output CSV file", required=False)
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline, model = load_model()
    data = pd.read_csv(args.input)
    X_prepared = pipeline.transform(data)
    predictions = model.predict(X_prepared)
    result = pd.DataFrame(predictions, columns=["predicted_price"])
    if args.output:
        result.to_csv(args.output, index=False)
        print(f"Predictions saved to {args.output}")
    else:
        print(result.head())


if __name__ == "__main__":
    main()
