import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.data import TensorDataset, DataLoader
import os

class RNNModel(nn.Module):
    def __init__(self, input_size, seq_length, hidden_size, output_size, num_layers=1, num_classes=3):
        super(RNNModel, self).__init__()
        self.input_size = input_size
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Define an RNN  layer
        self.rnn = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        
        # Define a fully connected layer that maps the hidden state output of RNN to the output size
        # Multiplied by num_classes as each output feature is classified into num_classes categories
        self.fc = nn.Linear(self.hidden_size, self.output_size * self.num_classes)
        

    def forward(self, x):
        # Forward pass through RNN
        # 
        # Input x shape: (batch, 1, seq_length * feature)
        # it should be (batch, seq_length, feature)
        # 
        # print("************", x.shape)
        # exit()
        # print(x.shape)
        past_days = self.seq_length
        batch_size, _, seq_length_times_feature = x.shape
        feature_size = seq_length_times_feature // past_days  # Calculate the feature size
        x = x.view(batch_size, past_days, feature_size)
        # print(x.shape)
        # exit()

        out, _ = self.rnn(x)
        # Process the output of the last time step
        out = self.fc(out[:, -1, :])
        
        # Reshape output to (batch_size, output_size, num_classes) for classification
        out = out.view(out.size(0), -1, self.num_classes)
        # print(out.shape)

        return out

class PositionalEncoding(nn.Module):
    '''
    This class is responsible for creating positional encodings 
    that are added to the embeddings in a Transformer model. 
    Positional encodings enable the model to understand the order or position of items in the input sequence, 
    which is crucial since the Transformer architecture itself doesn't have any mechanism 
    to recognize sequence order (like RNNs or CNNs do).

    Sinusoidal Encoding: The PositionalEncoding class generates positional encodings 
        based on sine and cosine functions of different frequencies. 
        For each position, it calculates a different sine value for even indices and 
        cosine value for odd indices in the encoding vector.
    '''
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)  # Dropout layer for regularization

        # Initialize positional encoding values
        position = torch.arange(max_len).unsqueeze(1)  # Shape: [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  # Shape: [d_model // 2]
        pe = torch.zeros(max_len, 1, d_model)  # Shape: [max_len, 1, d_model]
        pe[:, 0, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices; shape remains [max_len, 1, d_model]
        pe[:, 0, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices; shape remains [max_len, 1, d_model]
        self.register_buffer('pe', pe)  # Register pe as a constant buffer

    def forward(self, x):
        # Add the positional encoding to the input
        x = x + self.pe[:x.size(0)]  # Shape of x: [sequence_length, batch_size, d_model]
        return self.dropout(x)  # Apply dropout and return


class TransformerModel(nn.Module):
    '''
    Embedding: The nn.Linear layer is used for embedding the input features into a higher dimensional space (d_model). 
               This is a bit unconventional since embeddings are typically learned from an index 
               (like word embeddings in NLP), but in this case, 
               it seems the model expects continuous features which are linearly projected to the embedding space.

    Positional Encoding: Adds positional information to the embeddings. 
               The embeddings combined with these positional encodings are then passed to the encoder.

    Transformer Encoder: A stack of Transformer encoder layers (nn.TransformerEncoder) is used. 
               Each encoder layer typically consists of a multi-head self-attention mechanism 
               and a feedforward neural network, along with normalization and residual connections.
               It's important to note that when using PyTorch's built-in nn.TransformerEncoderLayer, 
               both normalization and residual connections are already incorporated internally.

    Output Layer: The final output for each sequence is taken from the last token's output of the Transformer encoder, 
               which is then passed through a linear layer (self.fc_out) to map it to the desired output size. 
               The use of the last token's output suggests that the model's task might involve classifying 
               or summarizing the entire sequence.

    Sequence Length Adjustment: The code includes commented-out sections for 
              adjusting the input sequence length to max_seq_length by 
              either truncating longer sequences or padding shorter ones. 
              This is crucial in batch processing where all input sequences must have the same length.
    '''
    def __init__(self, input_size, seq_length, output_size, num_classes, d_model, nhead, num_encoder_layers, dim_feedforward, max_seq_length):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Linear(input_size, d_model)  
        # Embedding layer; output shape: [batch_size, seq_len, d_model]
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_length)  
        # Positional encoding layer
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)  
        # Single transformer encoder layer
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)  
        # Stack of encoder layers
        self.fc_out = nn.Linear(d_model, output_size * num_classes)  
        # Final linear layer; output shape: [batch_size, output_size * num_classes]
        self.d_model = d_model
        self.output_size = output_size
        self.num_classes = num_classes
        self.input_size = input_size
        self.seq_length = seq_length
        self.max_seq_length = max_seq_length

    def forward(self, src):
        # Input x shape: (batch, 1, seq_length * feature)
        # it should be (batch, seq_length, feature)
        # print(src.shape)  
        past_days = self.seq_length
        batch_size, _, seq_length_times_feature = src.shape
        feature_size = seq_length_times_feature // past_days  # Calculate the feature size
        src = src.view(batch_size, past_days, feature_size)
        # print(src.shape)
        # exit()
        
        # Adjust sequence length for max_seq_length
        # if src.size(1) > self.max_seq_length:
        #     src = src[:, :self.max_seq_length, :]  # Truncate
        # elif src.size(1) < self.max_seq_length:
        #     # Pad with zeros
        #     padding_size = self.max_seq_length - src.size(1)
        #     padding = torch.zeros(batch_size, padding_size, feature_size, device=src.device)
        #     src = torch.cat([src, padding], dim=1)

        # 
        src = self.embedding(src) * math.sqrt(self.d_model)  
        # Shape after embedding: [batch_size, seq_len, d_model]
        src = self.pos_encoder(src)  
        # Shape after positional encoding: [seq_len, batch_size, d_model]
        output = self.transformer_encoder(src)  
        # Shape after transformer: [seq_len, batch_size, d_model]
        output = self.fc_out(output[:, -1, :])  
        # Select last token; shape: [batch_size, output_size * num_classes]

        output = output.view(output.size(0), -1, self.num_classes)  
        # Reshape; shape: [batch_size, output_size, num_classes]
        return output


def create_rnn_model(input_size, seq_length, hidden_size, output_size, num_layers=1, num_classes=3):
    # Create an RNN model instance with the specified parameters
    model = RNNModel(input_size, seq_length, hidden_size, output_size, num_layers, num_classes)
    print(f"RNN Model created with input_size={input_size}, seq_length = {seq_length}, hidden_size={hidden_size}, output_size={output_size}, num_layers={num_layers}, num_classes={num_classes}")
    return model

def create_transformer_model(input_size, seq_length, output_size, num_classes, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, max_seq_length=100):
    # Create a Transformer model instance with the specified parameters
    model = TransformerModel(input_size, seq_length, output_size, num_classes, d_model, nhead, num_layers, dim_feedforward, max_seq_length)
    print(f"Transformer Model created with input_size={input_size}, seq_length = {seq_length}, output_size={output_size}, num_classes={num_classes}, d_model={d_model}, nhead={nhead}, num_layers={num_layers}, dim_feedforward={dim_feedforward}, max_seq_length={max_seq_length}")
    return model




def train_model(model, X_train, y_train, X_test, y_test, epochs, batch_size, num_classes, criterion, optimizer, device):
    """
    Trains a PyTorch model and evaluates it on a test dataset.

    :param model: The PyTorch model to train.
    :param X_train: Training dataset features (inputs).
    :param y_train: Training dataset labels (targets).
    :param X_test: Test dataset features.
    :param y_test: Test dataset labels.
    :param epochs: Number of full passes through the training dataset.
    :param batch_size: Number of samples per batch to load.
    :param num_classes: Number of classes in the output layer.
    :param criterion: Loss function used for training.
    :param optimizer: Optimization algorithm.
    :param device: Device to run the training on ("cuda" or "cpu").

    :return: Lists containing the training and testing loss and accuracy for each epoch.
    """

    # Move the model to the specified device (GPU or CPU)
    model.to(device)
    print(f"Model moved to {device}")

    # Prepare the data loaders for training and testing datasets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize lists to keep track of metrics
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []

    # Start the training process
    for epoch in range(epochs):
        # Set the model to training mode (enables features like dropout)
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Iterate over batches of training data
        for inputs, labels in train_loader:
            # Move data to the specified device
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.long()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass: compute the model output
            outputs = model(inputs)
            outputs = outputs.view(-1, num_classes)
            labels = labels.view(-1)

            # Compute loss and perform a backward pass
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Calculate average loss and accuracy for this epoch
        avg_train_loss = running_loss / len(train_loader.dataset)
        avg_train_acc = correct / total
        train_loss.append(avg_train_loss)
        train_accuracy.append(avg_train_acc)

        print(f"Epoch [{epoch+1}/{epochs}]: Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")

        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.long()

                # Compute model output and loss
                outputs = model(inputs)
                outputs = outputs.view(-1, num_classes)
                labels = labels.view(-1)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate average loss and accuracy for this epoch on the test set
        avg_test_loss = running_loss / len(test_loader.dataset)
        avg_test_acc = correct / total
        test_loss.append(avg_test_loss)
        test_accuracy.append(avg_test_acc)

        print(f"Epoch [{epoch+1}/{epochs}]: Test Loss: {avg_test_loss:.4f}, Test Acc: {avg_test_acc:.4f}")

    return train_loss, train_accuracy, test_loss, test_accuracy





def train_and_save_model(model, X_train, y_train, X_test, y_test, epochs, batch_size, num_classes, criterion, optimizer, device, save_path):
    """
    Trains a PyTorch model and evaluates it on a test dataset, with added functionality to save the model.
    ...
    :param save_path: Directory path to save model checkpoints.
    """

    model.to(device)
    # print(f"Model moved to {device}")

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []

    best_test_acc = 0.0  # Initialize best accuracy for saving best model

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.long()
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.view(-1, num_classes)
            labels = labels.view(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_train_loss = running_loss / len(train_loader.dataset)
        avg_train_acc = correct / total
        train_loss.append(avg_train_loss)
        train_accuracy.append(avg_train_acc)

        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.long()
                outputs = model(inputs)
                outputs = outputs.view(-1, num_classes)
                labels = labels.view(-1)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_test_loss = running_loss / len(test_loader.dataset)
        avg_test_acc = correct / total
        test_loss.append(avg_test_loss)
        test_accuracy.append(avg_test_acc)

        print(f"Epoch [{epoch+1}/{epochs}]: Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, Test Loss: {avg_test_loss:.4f}, Test Acc: {avg_test_acc:.4f}")

        # Save model after each epoch
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), os.path.join(save_path, f'model_epoch_{epoch+1}_test_acc_{avg_test_acc:.4f}.pt'))

        # Optionally, save the best model only
        if avg_test_acc > best_test_acc:
            best_test_acc = avg_test_acc
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pt'))

    # Optionally, save the final model
    torch.save(model.state_dict(), os.path.join(save_path, 'final_model.pt'))

    return train_loss, train_accuracy, test_loss, test_accuracy



def test_model(model, X_test, y_test, batch_size, num_classes, criterion, device):
    """
    Evaluates a PyTorch model on a test dataset.

    :param model: The PyTorch model to evaluate.
    :param X_test: Test dataset features.
    :param y_test: Test dataset labels.
    :param batch_size: Number of samples per batch to load.
    :param num_classes: Number of classes in the output layer.
    :param criterion: Loss function used for evaluation.
    :param device: Device to run the evaluation on ("cuda" or "cpu").
    
    :return: Dictionary containing the test loss and accuracy.
    """

    # Load the model to the specified device
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    # Prepare the test loader
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.long()

            # Compute model output
            outputs = model(inputs)
            outputs = outputs.view(-1, num_classes)
            labels = labels.view(-1)

            # Compute loss
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_test_loss = test_loss / len(test_loader.dataset)
    avg_test_acc = correct / total

    print(f"Test Loss: {avg_test_loss:.4f}, Test Acc: {avg_test_acc:.4f}")

    return avg_test_loss, avg_test_acc





