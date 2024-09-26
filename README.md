# Federated Fraud Detection System

This project was created for my final year at university. implements a federated learning system for credit card fraud detection using the Flower framework and scikit-learn. The system is designed to train a Random Forest Classifier across multiple clients while maintaining data privacy.
## Project Structure
The project consists of three main Python files:

1. `client.py`: Implements the Flower client for local model training and evaluation.
2. `server.py`: Sets up the Flower server for coordinating the federated learning process.
3. `utils.py`: Contains utility functions for data processing, model parameter handling, and metric computation.

## Features

- Federated learning implementation using Flower framework
- Random Forest Classifier for fraud detection
- Data preprocessing and scaling
- SMOTE oversampling for handling class imbalance
- Multiple evaluation metrics (accuracy, precision, recall, F1-score, AUROC, AUPRC)
- Support for multiple clients (default: 3)

## Requirements

- Python 3.x
- pandas
- scikit-learn
- imbalanced-learn
- numpy
- flwr (Flower)

## Data

This project uses the Credit Card Fraud Detection dataset. Due to [size constraints/data sensitivity], the full dataset is not included in this repository. 

To use this project:
1. Download the dataset from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud.
2. Place the 'creditcard.csv' file in the project root directory

Note: Ensure you have appropriate permissions to use this dataset.

Data description:
- File name: creditcard.csv
- Features: [List main features]
- Target variable: 'Class' (1 for fraudulent transactions, 0 for normal)
## Usage

1. Start the server:
   ```python server.py```
2. Run the clients (in separate terminals):
   ```python client.py```

Repeat this step for each client, ```python client1.py```, then ```python client2.py```, (default: 3 clients).
