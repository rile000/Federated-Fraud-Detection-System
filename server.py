import flwr as fl
from sklearn.ensemble import RandomForestClassifier
import utils
from typing import Dict

def send_round_number(current_round: int) -> Dict:
    """Send the current round number to the client."""
    return {"current_round": current_round}

def aggregate_metrics(client_results) -> Dict:
    if not client_results:
        return {}
    else:
        total_samples = 0
        # Initialize aggregated metrics
        combined_metrics = {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "auroc": 0,
            "auprc": 0,
        }
        
        # Aggregate metrics weighted by the number of samples
        for num_samples, metrics in client_results:
            for metric_name, metric_value in metrics.items():
                if metric_name in combined_metrics:
                    combined_metrics[metric_name] += metric_value * num_samples
            total_samples += num_samples
        
        # Calculate weighted averages
        for metric_name in combined_metrics.keys():
            combined_metrics[metric_name] /= total_samples
        
        return combined_metrics

if __name__ == "__main__":
    num_clients = 3
    num_rounds = 2
    # Define the model
    model = RandomForestClassifier()
    # Define the strategy with the new aggregation function
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=num_clients,
        evaluate_metrics_aggregation_fn=aggregate_metrics,
        fit_metrics_aggregation_fn=aggregate_metrics,
        on_fit_config_fn=send_round_number
    )
    # Start the Flower server
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy
    )