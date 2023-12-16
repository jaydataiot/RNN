from sklearn.model_selection import train_test_split
import torch
import pandas as pd


def preprocess_data(df, past_days=1, future_days=1, feature_columns=None, target_columns=None):
    """
    Preprocess the data  for prediction tasks, suitable for both RNN and Transformer models.

    This function transforms a time-series DataFrame into a format that can be used for training
    machine learning models, particularly for sequence prediction tasks. It creates feature and target
    datasets based on specified past and future day windows. 
    When there's missing data in the past p [or past_days] days, we simply repeat the most recent day's data. 
    This is done by dynamically filling in missing data for each sequence where necessary.
       
    :param df: DataFrame with the original data.
    :param past_days: Number of past days to consider for creating features (default 1).
                      The data from these days will be used as input features for the model.
    :param future_days: Number of future days to predict (default 1).
                        The data from these days will be used as targets for the model.
    :param feature_columns: List of columns to use as features. If None, all columns except 'date' are used.
                            These columns are used to extract input data.
    :param target_columns: List of columns to predict. If None, all columns except 'date' are used.
                           These columns are the ones that the model will attempt to predict.
    
    :return: A tuple of two DataFrames (features, targets).
             The 'features' DataFrame contains input data structured in a way where each row represents
             the concatenated values of feature columns for a series of past days.
             The 'targets' DataFrame contains the data to be predicted, structured similarly with
             each row representing the future values for a series of future days.
    """
    
    # Set default feature and target columns if not specified
    if feature_columns is None:
        feature_columns = df.columns.drop('date')
    if target_columns is None:
        target_columns = df.columns.drop('date')

    # Convert 'date' column to datetime type for consistency
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    
    # Create a date range for each set of feature sequences, fill missing data dynamically
    features = []
    targets = []
    for start_idx in range(len(df) - past_days - future_days + 1):
        end_idx = start_idx + past_days
        # Define the date range for the current sequence of past days (features)
        feature_range = pd.date_range(start=df.index[start_idx], periods=past_days)
        # Define the date range for the current sequence of future days (targets)
        target_range = pd.date_range(start=df.index[end_idx], periods=future_days)

        # Select and flatten feature data, filling missing dates with the last available data (forward fill)
        # feature_data = df.reindex(feature_range).fillna(method='ffill')[feature_columns].values.flatten().tolist()
        feature_data = df.reindex(feature_range).ffill()[feature_columns].values.flatten().tolist()
        # Select and flatten target data similarly
        # target_data = df.reindex(target_range).fillna(method='ffill')[target_columns].values.flatten().tolist()
        target_data = df.reindex(target_range).ffill()[target_columns].values.flatten().tolist()

        
        # Add the processed data to the lists
        features.append(feature_data)
        targets.append(target_data)

    # Create column names for features and targets based on the time offsets and sequence lengths
    flat_feature_columns = [f'{col}_t-{i}' for i in range(past_days, 0, -1) for col in feature_columns]
    flat_target_columns = [f'{col}_t+{i}' for i in range(1, future_days + 1) for col in target_columns]

    # Convert the lists of features and targets into DataFrames
    features_df = pd.DataFrame(features, columns=flat_feature_columns)
    targets_df = pd.DataFrame(targets, columns=flat_target_columns)

    return features_df, targets_df



def split_data(features_df, targets_df, train_size=0.8, batch_size=None):
    """
    Split the data into training and testing sets, and optionally ensures divisibility by batch size.
    Also encodes categorical data to numeric values.

    :param features_df: DataFrame containing the feature data.
    :param targets_df: DataFrame containing the target data.
    :param train_size: Proportion of the dataset to include in the train split.
    :param batch_size: Batch size for model training. If provided, trims datasets to be divisible by batch size.
    :return: Training and testing data as PyTorch tensors.
    """

    # Encode categories (like 'L', 'M', 'H') to numeric values (0, 1, 2) before splitting
    mapping = {'L': 0, 'M': 1, 'H': 2}
    features_df_encoded = features_df.replace(mapping).astype(int)
    targets_df_encoded = targets_df.replace(mapping).astype(int)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features_df_encoded, targets_df_encoded, train_size=train_size, random_state=42
    )

    # If a batch size is provided, adjust the lengths of training and testing sets to be divisible by the batch size
    if batch_size is not None:
        train_len = (len(X_train) // batch_size) * batch_size
        test_len = (len(X_test) // batch_size) * batch_size
        X_train, y_train = X_train.iloc[:train_len], y_train.iloc[:train_len]
        X_test, y_test = X_test.iloc[:test_len], y_test.iloc[:test_len]

    # Convert the training and testing sets to PyTorch tensors and add an extra dimension for batch processing
    X_train_tensor = torch.tensor(X_train.values).float().unsqueeze(1)  # Add a dimension of size 1 at the 1st index
    X_test_tensor = torch.tensor(X_test.values).float().unsqueeze(1)
    y_train_tensor = torch.tensor(y_train.values).float()
    y_test_tensor = torch.tensor(y_test.values).float()

    # Return the tensors
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor






