from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

def evaluate(model: torch.nn.Module, test_loader: DataLoader, device: torch.device) -> None:
    """
    Evaluates the trained model on the test dataset and plots the ROC-AUC curve.

    Args:
        model (torch.nn.Module): The trained neural network model.
        test_loader (DataLoader): Dataloader for the test set.
        device (torch.device): Device to run the model on (CPU or GPU).
    """
    model.eval()  # Set the model to evaluation mode
    y_true = []
    y_probs = []  # We will store predicted probabilities here

    # Inference
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            y_true.extend(labels.cpu().numpy())
            y_probs.extend(outputs.cpu().numpy())  # Collect predicted probabilities

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    # Calculate ROC-AUC score
    roc_auc = roc_auc_score(y_true, y_probs)

    # Binarize probabilities (round them)
    y_pred = (y_probs >= 0.5).astype(int)

    # Calculate accuracy and classification report
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    # Generate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
