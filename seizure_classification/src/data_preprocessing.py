import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from typing import Tuple

def load_data(file_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads the Epileptic Seizure dataset from a CSV file and prepares features (X) and labels (y).

    Args:
        file_path (str): Path to the dataset file.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Feature matrix X and label vector y.
    """
    data = pd.read_csv(file_path)
    data = data.drop(columns=['Unnamed'])
    data['y'] = data['y'].apply(lambda x: 1 if x == 1 else 0)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return X, y

def preprocess_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, val_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, ...]:
    """
    Scales and splits the data into training, validation, and test sets.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Label vector.
        test_size (float): Proportion of the data for testing.
        val_size (float): Proportion of the training data for validation.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple[np.ndarray, ...]: Scaled training, validation, and test sets.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train_val, X_test, y_train_val, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size, random_state=random_state, stratify=y_train_val)

    return X_train, X_val, X_test, y_train, y_val, y_test

def convert_to_tensors(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, y_train: pd.Series, y_val: pd.Series, y_test: pd.Series, device: torch.device) -> Tuple[torch.Tensor, ...]:
    """
    Converts numpy arrays into PyTorch tensors and transfers them to the specified device.

    Args:
        X_train (np.ndarray): Training features.
        X_val (np.ndarray): Validation features.
        X_test (np.ndarray): Test features.
        y_train (pd.Series): Training labels.
        y_val (pd.Series): Validation labels.
        y_test (pd.Series): Test labels.
        device (torch.device): Device to transfer the tensors to (CPU or GPU).

    Returns:
        Tuple[torch.Tensor, ...]: Tensors for training, validation, and test sets.
    """
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1).to(device)

    return X_train_tensor, X_val_tensor, X_test_tensor, y_train_tensor, y_val_tensor, y_test_tensor
