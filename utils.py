import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from typing import List
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

def load_data(file_path):
    # Load the dataset from the CSV file
    data = pd.read_csv(file_path)
    # Separate features and target variable
    X = data.drop('Class', axis=1)
    y = data['Class']
    return X, y

def preprocess_data(X):
    # Scale the Time and Amount features
    scaler = StandardScaler()
    X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])
    return X

def split_data(X, y, num_clients):
    # Split the data into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Split the training data into equal subsets for each client
    X_train_splits = np.array_split(X_train, num_clients)
    y_train_splits = np.array_split(y_train, num_clients)

    # Split the validation data into equal subsets for each client
    X_val_splits = np.array_split(X_val, num_clients)
    y_val_splits = np.array_split(y_val, num_clients)

    # Split the test data into equal subsets for each client
    X_test_splits = np.array_split(X_test, num_clients)
    y_test_splits = np.array_split(y_test, num_clients)

    return X_train_splits, y_train_splits, X_val_splits, y_val_splits, X_test_splits, y_test_splits

def get_client_data(client_id):
    # Load the dataset
    X, y = load_data(r'E:\Diss\validatied federated\creditcard.csv')

    # Preprocess the data (scaling)
    X_preprocessed = preprocess_data(X)

    # Split the preprocessed data between the desired number of clients
    num_clients = 3
    X_train_splits, y_train_splits, X_val_splits, y_val_splits, X_test_splits, y_test_splits = split_data(X_preprocessed, y, num_clients)

    return X_train_splits[client_id - 1], y_train_splits[client_id - 1], X_val_splits[client_id - 1], y_val_splits[client_id - 1], X_test_splits[client_id - 1], y_test_splits[client_id - 1]

def set_params(model: RandomForestClassifier, params: List[np.ndarray]) -> RandomForestClassifier:
    if model is None:
        model = RandomForestClassifier()

    model.n_estimators = int(params[0])
    model.max_depth = None if int(params[1]) == -1 else int(params[1])
    model.min_samples_split = int(params[2])
    model.min_samples_leaf = int(params[3])
    return model

def get_params(model: RandomForestClassifier) -> List[np.ndarray]:
    if model is None:
        # Return a valid set of default parameters
        return [
            np.array(100, dtype=int),  # n_estimators
            np.array(-1, dtype=int),   # max_depth (None)
            np.array(2, dtype=int),    # min_samples_split
            np.array(1, dtype=int)     # min_samples_leaf
        ]

    params = [
        np.array(model.n_estimators, dtype=int),
        np.array(model.max_depth, dtype=int) if model.max_depth is not None else np.array(-1, dtype=int),
        np.array(model.min_samples_split, dtype=int),
        np.array(model.min_samples_leaf, dtype=int),
    ]
    return params

def oversample_train_data(X_train, y_train):
    # Apply SMOTE to balance the classes in the training set
    smote = SMOTE(random_state=42)
    X_train_oversampled, y_train_oversampled = smote.fit_resample(X_train, y_train)
    return X_train_oversampled, y_train_oversampled



def compute_metrics(y_true, y_pred, y_pred_proba):
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    auroc = roc_auc_score(y_true, y_pred_proba)
    auprc = average_precision_score(y_true, y_pred_proba)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auroc': auroc,
        'auprc': auprc
    }