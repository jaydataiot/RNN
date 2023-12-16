import torch
import torch.nn as nn
import torch.optim as optim

config = {
    "data": {
        "past_days": 5,  # Number of past days' data to use as input
        "future_days": 1,  # Number of future days to predict
                        # ['VIX', 'WIL', 'Spread', 'Oil', 'RC']
        "feature_columns": ['VIX', 'WIL', 'Spread', 'Oil'],  # Columns to be used as features
        "target_columns": ['RC']  # Columns to be predicted as targets
    },
    "model_common": {
        "batch_size": 8,  # Size of each training batch
        # Dynamically calculate input and output sizes
        # "input_size": lambda config: len(config["data"]["feature_columns"]) * config["data"]["past_days"],
        "input_size": lambda config: len(config["data"]["feature_columns"]),
        "output_size": lambda config: len(config["data"]["target_columns"]) * config["data"]["future_days"],
        "num_classes": 3  # Number of classes for each output feature
    },
    "rnn": {
        "num_layers": 4,  # Number of layers in  the RNN model
        "hidden_size": 256,  # Size of the hidden layers in the RNN model
        "optimizer": {
            "type": "Adam",
            "learning_rate": 0.0001
        }
    },
    "transformer": {
        "d_model": 512,  # The number of expected features in the transformer (input dimension)
        "nhead": 8,  # Number of heads in the multiheadattention model
        "num_layers": 6,  # Number of sub-encoder-layers in the transformer encoder
        "dim_feedforward": 2048,  # Dimension of the feedforward network model in the transformer
        "max_seq_length": 100,  # Maximum sequence length for positional encodings
        "optimizer": {
            "type": "Adam",
            "learning_rate": 0.00001
        }
    },
    "training": {
        "train_size": 0.7,
        "num_epochs": 100,  # Total number of training epochs
        # "device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
        "device": "cpu",
        "loss_function": "CrossEntropyLoss"  # Type of the loss function (to be instantiated in the training script)
    }
}

if __name__ == "__main__":
    # Assigning values to parameters
    past_days = config['data']['past_days']
    future_days = config['data']['future_days']
    feature_columns = config['data']['feature_columns']
    target_columns = config['data']['target_columns']

    batch_size = config['model_common']['batch_size']
    input_size = config['model_common']['input_size'](config)  # Note the function call
    output_size = config['model_common']['output_size'](config)  # Note the function call
    num_classes = config['model_common']['num_classes']

    num_layers_rnn = config['rnn']['num_layers']
    hidden_size_rnn = config['rnn']['hidden_size']

    d_model = config['transformer']['d_model']
    nhead = config['transformer']['nhead']
    num_layers_transformer = config['transformer']['num_layers']
    dim_feedforward = config['transformer']['dim_feedforward']
    max_seq_length = config['transformer']['max_seq_length']

    train_size = config['training']['train_size']
    num_epochs = config['training']['num_epochs']
    device = config['training']['device']
    loss_function_name = config['training']['loss_function']

    # Instantiate loss function
    if loss_function_name == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    else:
        # Default or other loss functions
        criterion = nn.MSELoss()  # Example, modify as needed

    
