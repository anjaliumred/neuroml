import torch
from torch.utils.data import DataLoader, TensorDataset
from src.data_preprocessing import load_data, preprocess_data, convert_to_tensors
from src.model import SeizureNN
from src.train import train_with_early_stopping
from src.evaluate import evaluate

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FILE_PATH = r""
BEST_MODEL_PATH = r""
num_epochs = 100
batch_size = 64
early_stopping_patience = 10

def main() -> None:
    """
    Main function to run the training and evaluation of the neural network model.
    """
    # 1. Load and preprocess data
    X, y = load_data(FILE_PATH)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(X, y)
    X_train_tensor, X_val_tensor, X_test_tensor, y_train_tensor, y_val_tensor, y_test_tensor = convert_to_tensors(X_train, X_val, X_test, y_train, y_val, y_test, device)

    # 2. Create dataloaders
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

    # 3. Initialize model, loss function, and optimizer
    model = SeizureNN().to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 4. Train the model with early stopping
    train_with_early_stopping(model, criterion, optimizer, train_loader, val_loader, num_epochs, early_stopping_patience, device, BEST_MODEL_PATH)

    # 5. Evaluate the model on test set
    evaluate(model, test_loader, device)

if __name__ == "__main__":
    main()
