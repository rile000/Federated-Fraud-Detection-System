Federated Fraud Detection System
This project implements a federated learning system for credit card fraud detection using the Flower framework and scikit-learn. The system is designed to train a Random Forest Classifier across multiple clients while maintaining data privacy.
Project Structure
The project consists of three main Python files:

client.py: Implements the Flower client for local model training and evaluation.
server.py: Sets up the Flower server for coordinating the federated learning process.
utils.py: Contains utility functions for data processing, model parameter handling, and metric computation.

Features

Federated learning implementation using Flower framework
Random Forest Classifier for fraud detection
Data preprocessing and scaling
SMOTE oversampling for handling class imbalance
Multiple evaluation metrics (accuracy, precision, recall, F1-score, AUROC, AUPRC)
Support for multiple clients (default: 3)

Requirements

Python 3.x
pandas
scikit-learn
imbalanced-learn
numpy
flwr (Flower)

Dataset
The project uses a credit card transaction dataset (creditcard.csv) for fraud detection. Ensure that this file is placed in the correct directory as specified in the utils.py file.
Usage

Start the server:
