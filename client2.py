import flwr as fl
from sklearn.ensemble import RandomForestClassifier
import utils
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id, model_params):
        super().__init__()
        self.client_id = client_id
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = utils.get_client_data(client_id)
        self.model = None
        self.model_params = model_params
        self.eval_fn = self.get_eval_fn(self.X_val, self.y_val)

    def get_eval_fn(self, X_val, y_val):
        # Define the evaluation function
        def evaluate(model, X_val, y_val):
            # Evaluate the model on the validation set
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]  # Get the probability of the positive class
            metrics = utils.compute_metrics(y_val, y_pred, y_pred_proba)
            return metrics
        return lambda model: evaluate(model, X_val, y_val)

    # Get the current local model parameters
    def get_parameters(self, config):
        return utils.get_params(self.model)

    def fit(self, parameters, config):
        self.model = utils.set_params(RandomForestClassifier(**self.model_params), parameters)
        
        # Apply oversampling to the training data
        X_train_oversampled, y_train_oversampled = utils.oversample_train_data(self.X_train, self.y_train)
        
        # Fit the model with the oversampled training data
        self.model.fit(X_train_oversampled, y_train_oversampled)
        
        trained_params = utils.get_params(self.model)
        # Store the fitted model instance
        self.model = self.model
        
        return trained_params, len(X_train_oversampled), {}

    # Evaluate the local model, return the evaluation result to the server
    def evaluate(self, parameters, config):
        utils.set_params(self.model, parameters)
        if self.model is not None:
            y_pred = self.model.predict(self.X_val)
            y_pred_proba = self.model.predict_proba(self.X_val)[:, 1]  # Get the probability of the positive class
            metrics = utils.compute_metrics(self.y_val, y_pred, y_pred_proba)
            loss = 0.0
            return loss, len(self.X_val), metrics
        else:
            # Return dummy values if the model is not fitted
            loss = 0.0
            num_examples = len(self.X_val)
            metrics = {}
            return loss, num_examples, metrics

if __name__ == "__main__":
    client_id = 2
    model_params = {
        'n_estimators': 300,
        'max_depth': 5,
        'min_samples_split': 2,
        'min_samples_leaf': 1
    }
    print(f"Client {client_id}:\n")
    client = FlowerClient(client_id, model_params)
    fl.client.start_client(server_address="127.0.0.1:8080", client=client.to_client())