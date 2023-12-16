import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import os

from config import config
from data import preprocess_data, split_data
from model import RNNModel, TransformerModel, create_rnn_model, create_transformer_model
from model import train_model, train_and_save_model
from model import test_model

def train_pipeline(config, file_path, model_type, save_path):
    print("\n=================================== Load data:")
    # The dataset is then loaded into a pandas DataFrame.
    # Uncomment the second line to  load only the first 100 rows of the dataset.
    data = pd.read_csv(file_path)
    # data = pd.read_csv(file_path, nrows=100)

    # Display the first few rows of the dataframe
    # Printing the head of the DataFrame gives us a quick overview of the dataset,
    # including column names and some initial data to understand its structure.
    print("First few rows of the dataset:")
    print(data.head())

    # Display the shape (dimensions) of the dataframe
    # This print statement will show the number of rows and columns in the DataFrame,
    # which is helpful to understand the size of the dataset.
    print("\nShape of the dataset (rows, columns):")
    print(data.shape)
    
    past_days = config['data']['past_days']
    future_days = config['data']['future_days']
    feature_columns = config['data']['feature_columns']
    target_columns = config['data']['target_columns']

    features_df, targets_df = preprocess_data(data, past_days, future_days, feature_columns, target_columns)

    print("\nShape of features_df and targets_df:")
    print(features_df.shape, targets_df.shape)
    # exit()
    # 


    # Split data
    print("\n=================================== Split data:")
    train_size = config['training']['train_size']
    batch_size = config['model_common']['batch_size']
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = split_data(features_df, targets_df, train_size, batch_size)

    # Model selection and instantiation
    print("\n=================================== Load config and create model:")
    # model_type = model_type
    input_size = config['model_common']['input_size'](config)
    output_size = config['model_common']['output_size'](config)
    num_classes = config['model_common']['num_classes']
    seq_length = config['data']['past_days']

    # Model creation based on the specified type (RNN or Transformer)
    if model_type == 'rnn':
        num_layers_rnn = config['rnn']['num_layers']
        hidden_size_rnn = config['rnn']['hidden_size']
        model = create_rnn_model(input_size, seq_length, hidden_size_rnn, output_size, num_layers_rnn, num_classes)
        optimizer_config = config['rnn']['optimizer']
    elif model_type == 'transformer':
        d_model = config['transformer']['d_model']
        nhead = config['transformer']['nhead']
        num_layers_transformer = config['transformer']['num_layers']
        dim_feedforward = config['transformer']['dim_feedforward']
        max_seq_length = config['transformer']['max_seq_length']
        model = create_transformer_model(input_size, seq_length, output_size, num_classes, d_model, nhead, num_layers_transformer, dim_feedforward, max_seq_length)
        optimizer_config = config['transformer']['optimizer']
    
    # Optimizer
    if optimizer_config['type'] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=optimizer_config['learning_rate'])

    # Loss function
    loss_function_name = config['training']['loss_function']
    if loss_function_name == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()  # Default or other loss functions can be specified

    print(model)

    # Training
    print("\n=================================== Train model:")
    num_epochs = config['training']['num_epochs']
    device = config['training']['device']
    # train_model(model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, num_epochs, batch_size, num_classes, criterion, optimizer, device)
    train_and_save_model(model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, num_epochs, batch_size, num_classes, criterion, optimizer, device, save_path)



def test_pipeline(config, file_path, model_type, model_path):
    print("\n=================================== Load data:")
    # Load the dataset
    data = pd.read_csv(file_path)
    print("First few rows of the dataset:")
    print(data.head())
    print("\nShape of the dataset (rows, columns):")
    print(data.shape)
    
    # Preprocess data
    past_days = config['data']['past_days']
    future_days = config['data']['future_days']
    feature_columns = config['data']['feature_columns']
    target_columns = config['data']['target_columns']

    features_df, targets_df = preprocess_data(data, past_days, future_days, feature_columns, target_columns)

    # Convert features and targets to tensors
    batch_size = config['model_common']['batch_size']
    input_size = config['model_common']['input_size'](config)  # Note the function call
    output_size = config['model_common']['output_size'](config)  # Note the function call
    num_classes = config['model_common']['num_classes']
    
    features_tensor, _, targets_tensor, _ = split_data(features_df, targets_df, 0.99999999, batch_size)

    print(features_tensor.shape, targets_tensor.shape)
    # Load model
    print("\n=================================== Load model:")
    device = config['training']['device']
    seq_length = config['data']['past_days']

    if model_type == 'rnn':
        num_layers_rnn = config['rnn']['num_layers']
        hidden_size_rnn = config['rnn']['hidden_size']
        # model = RNNModel(input_size, hidden_size_rnn, output_size, num_layers_rnn, num_classes)
        model = create_rnn_model(input_size, seq_length, hidden_size_rnn, output_size, num_layers_rnn, num_classes)
    elif model_type == 'transformer':
        d_model = config['transformer']['d_model']
        nhead = config['transformer']['nhead']
        num_layers_transformer = config['transformer']['num_layers']
        dim_feedforward = config['transformer']['dim_feedforward']
        max_seq_length = config['transformer']['max_seq_length']
        model = TransformerModel(input_size, seq_length, output_size, num_classes, d_model, nhead, num_layers_transformer, dim_feedforward, max_seq_length)

    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # Testing
    print("\n=================================== Test model:")
    criterion = nn.CrossEntropyLoss()  # Assuming CrossEntropyLoss for simplicity
    test_loss, test_accuracy = test_model(model, features_tensor, targets_tensor, batch_size, num_classes, criterion, device)

    # print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


# ==========================  Usage
# The script then demonstrates how to use these functions for training and testing a model. 
# It allows the user to specify the model type (rnn or transformer), the mode (train or test), 
# and paths for data and saved models. 
# This setup makes it easy to experiment with different models and configurations.
# ========================== 

time_range = 'before'
#time_range = 'during'

#model_type = 'rnn'
model_type = 'transformer'

mode = 'train'
if mode == 'train':
    file_path = 'data/d_n_' + time_range + '_train.csv'
    save_path = 'model_checkpoints/'+ model_type + '/' + time_range
    train_pipeline(config, file_path, model_type, save_path)

mode = 'test'
if mode == 'test':
    file_path = 'data/d_n_' + time_range + '_test.csv'
    model_path = 'model_checkpoints/' + model_type + '/' + time_range + '/final_model.pt'
    test_pipeline(config, file_path, model_type, model_path)






