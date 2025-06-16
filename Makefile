.PHONY: data train evaluate all

data:
	python src/make_dataset.py

train: data
	python src/train.py

evaluate: train
	python src/evaluate.py

all: evaluate
