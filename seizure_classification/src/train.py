import torch
from torch.utils.data import DataLoader

def train_with_early_stopping(model: torch.nn.Module, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int, early_stopping_patience: int, device: torch.device, best_model_path: str) -> None:
    """
    Trains the neural network with early stopping based on validation loss.

    Args:
        model (torch.nn.Module): The neural network model.
        criterion (torch.nn.Module): Loss function (e.g., BCELoss).
        optimizer (torch.optim.Optimizer): Optimizer (e.g., Adam).
        train_loader (DataLoader): Dataloader for training set.
        val_loader (DataLoader): Dataloader for validation set.
        num_epochs (int): Maximum number of training epochs.
        early_stopping_patience (int): Number of epochs with no improvement before stopping.
        device (torch.device): Device to run the model on (CPU or GPU).
    """
    min_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/len(train_loader):.4f}, Validation Loss: {avg_val_loss:.4f}")

        # Early stopping logic
        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Validation loss improved to {avg_val_loss:.4f}. Model saved.")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break
