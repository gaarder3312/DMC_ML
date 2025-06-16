# House Price Prediction

This project demonstrates a simple machine learning engineering workflow for predicting house prices using the California Housing dataset. The code is based on the notebook located in `notebooks/house_price_prediction.ipynb` and has been converted into reusable scripts.

## Project Structure

```
├── data
│   ├── raw            # Original downloaded dataset
│   ├── processed      # Train/Test splits
│   └── scores         # Evaluation metrics
├── models             # Trained models
├── notebooks          # Jupyter notebooks
├── src                # Source code for the pipeline
├── requirements.txt   # Python dependencies
└── setup.py           # Makes project installable
```

## Setup

1. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Install the package in editable mode:

```bash
pip install -e .
```

## Usage

The entire process can be executed with the following commands:

```bash
python src/make_dataset.py       # Download and prepare data
python src/train.py              # Train the model
python src/evaluate.py           # Evaluate the model
```

To make predictions on new data:

```bash
python src/predict.py --input path/to/data.csv --output predictions.csv
```

## Automation

A `Makefile` is provided to automate the workflow:

```bash
make all     # runs data preparation, training and evaluation
```

## License

This project is licensed under the MIT License.
