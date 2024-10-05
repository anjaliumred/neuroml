import torch
import torch.nn as nn

class SeizureNN(nn.Module):
    """
    Neural network for Epileptic Seizure Classification.
    It consists of 3 fully connected layers with ReLU activations, followed by a sigmoid output layer.
    """
    def __init__(self) -> None:
        """
        Initializes the layers of the neural network.
        """
        super(SeizureNN, self).__init__()
        self.fc1 = nn.Linear(178, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of features.

        Returns:
            torch.Tensor: Output tensor with predicted probabilities.
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x
