"""
KANWrapper.py

This module provides the KANWrapper class, which encapsulates the Kolmogorov-Arnold Network (KAN) 
with additional functionalities for initialization, training, mutation, inheritance, and symbolic function fixing.

Dependencies:
- logging
- random
- typing
- tkinter
- torch
- matplotlib

Usage:
    from KANWrapper import KANWrapper
"""

from datetime import datetime
import logging
import random
from typing import Optional, List, Dict, Any, Tuple
from tkinter import filedialog, messagebox
import tkinter as tk
import torch
from torch.optim import Adam, SGD
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from kan import KAN  # Ensure this is the correct import from the pykan module
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from torch.utils.data import TensorDataset, DataLoader

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class KANWrapper(KAN):
    def __init__(
        self,
        width: List[int],
        grid: int = 3,
        k: int = 3,
        noise_scale: float = 0.1,
        noise_scale_base: float = 0.1,
        base_fun: torch.nn.Module = torch.nn.SiLU(),
        symbolic_enabled: bool = True,
        bias_trainable: bool = True,
        grid_eps: float = 1,
        grid_range: List[int] = [-1, 1],
        sp_trainable: bool = True,
        sb_trainable: bool = True,
        device: str = "cpu",
        seed: int = 0,
        learnable_params: Optional[Dict[str, Any]] = None,
        lr: float = 0.01,
        epochs: int = 100,
    ) -> None:
        """
        Initialize the Kolmogorov-Arnold Network (KAN).

        Args:
            width (List[int]): List of integers specifying the width of each layer.
            grid (int): Grid size for the KAN.
            k (int): Number of neurons in the hidden layer.
            noise_scale (float): Scale of the noise.
            noise_scale_base (float): Base scale of the noise.
            base_fun (torch.nn.Module): Base function for the KAN.
            symbolic_enabled (bool): Whether symbolic functions are enabled.
            bias_trainable (bool): Whether biases are trainable.
            grid_eps (float): Epsilon for the grid.
            grid_range (List[int]): Range for the grid.
            sp_trainable (bool): Whether sp is trainable.
            sb_trainable (bool): Whether sb is trainable.
            device (str): Device to run the model on.
            seed (int): Random seed.
            learnable_params (Optional[Dict[str, Any]]): Optional dictionary of additional learnable parameters.
            lr (float): Learning rate for the optimizer.
            epochs (int): Number of training epochs.
        """
        super().__init__(
            width=width,
            grid=grid,
            k=k,
            noise_scale=noise_scale,
            noise_scale_base=noise_scale_base,
            base_fun=base_fun,
            symbolic_enabled=symbolic_enabled,
            bias_trainable=bias_trainable,
            grid_eps=grid_eps,
            grid_range=grid_range,
            sp_trainable=sp_trainable,
            sb_trainable=sb_trainable,
            device=device,
            seed=seed,
        )

        self.epochs = epochs
        self.learnable_params = learnable_params or {}
        self.optimizer = Adam(self.parameters(), lr=lr)
        logging.debug(f"KANWrapper initialized with parameters: {self.__dict__}")

    def _initialize_parameters(self) -> None:
        """
        Initialize all model parameters and learnable parameters with log-normal distribution.
        """
        try:
            for param in self.parameters():
                param.data = torch.randn_like(param).log_normal_()
            for key, value in self.learnable_params.items():
                if isinstance(value, torch.Tensor):
                    self.learnable_params[key] = torch.randn_like(value).log_normal_()
                else:
                    self.learnable_params[key] = torch.tensor(value).log_normal_()
            logging.debug("All model parameters initialized.")
        except Exception as e:
            logging.error(f"Error initializing parameters: {e}")
            raise

    def _initialize_missing_keys(self) -> None:
        """
        Ensure all required keys are present in learnable_params and initialize them if missing.
        """
        try:
            required_keys = [name for name, _ in self.named_parameters()]
            for key in required_keys:
                if key not in self.learnable_params:
                    self.learnable_params[key] = torch.randn(1).log_normal_()
                    logging.debug(f"Initialized missing key {key}.")
            logging.debug("All missing keys initialized.")
        except Exception as e:
            logging.error(f"Error initializing missing keys: {e}")
            raise

    def save(self, filename: str) -> None:
        """
        Save the network's weights, biases, and learnable parameters to a file.

        Args:
            filename (str): The file path to save the network.
        """
        try:
            state = {
                "model_state_dict": self.state_dict(),
                "learnable_params": self.learnable_params,
                "optimizer_state_dict": self.optimizer.state_dict(),
            }
            with open(filename, "wb") as f:
                torch.save(state, f)
            logging.info(f"Model saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving model: {e}")
            raise

    @classmethod
    def load(cls, filename: str) -> "KANWrapper":
        """
        Load a network's weights, biases, and learnable parameters from a file.

        Args:
            filename (str): The file path to load the network from.

        Returns:
            KANWrapper: An instance of the KANWrapper class with loaded weights, biases, and learnable parameters.
        """
        try:
            with open(filename, "rb") as file:
                state = torch.load(file)
            instance = cls(
                width=state["learnable_params"]["width"],
                grid=state["learnable_params"]["grid"],
                k=state["learnable_params"]["k"],
            )
            instance.load_state_dict(state["model_state_dict"])
            instance.optimizer.load_state_dict(state["optimizer_state_dict"])
            instance.learnable_params = state["learnable_params"]
            logging.info(f"Model loaded from {filename}")
            return instance
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

    def fix_symbolic_function(
        self, l: int, i: int, j: int, expression: str, fit_parameters: bool = True
    ) -> None:
        """Fix a symbolic function in the KAN model."""
        try:
            self.fix_symbolic(l, i, j, expression, fit_parameters)
            logging.info(f"Fixed symbolic function {expression} at ({l}, {i}, {j})")
        except AttributeError as e:
            logging.error(f"Error in fix_symbolic_function: {e}")
            raise

    def get_activation_range(
        self, l: int, i: int, j: int
    ) -> Tuple[float, float, float, float]:
        """Get the activation range for a specific neuron."""
        try:
            x_min, x_max, y_min, y_max = self.get_range(l, i, j)
            logging.info(
                f"Activation range for ({l}, {i}, {j}): x=({x_min}, {x_max}), y=({y_min}, {y_max})"
            )
            return x_min, x_max, y_min, y_max
        except Exception as e:
            logging.error(f"Error getting activation range: {e}")
            raise

    def initialize_from_another_model(
        self, other: "KANWrapper", input_tensor: torch.Tensor
    ) -> None:
        """
        Initialize the model from another parent model.

        Args:
            other (KANWrapper): The parent model to initialize from.
            input_tensor (torch.Tensor): Input tensor for initialization.
        """
        try:
            if not isinstance(other, KANWrapper):
                raise TypeError("other must be an instance of KANWrapper")
            if not isinstance(input_tensor, torch.Tensor):
                raise TypeError("input_tensor must be a torch.Tensor")

            if input_tensor.dim() == 1:
                input_tensor = input_tensor.unsqueeze(0)

            self.initialize_from_another_model(other, input_tensor)
            logging.info("Initialized from another model")

            self._initialize_missing_keys()

            if messagebox.askyesno(
                "Save Initialized Model",
                "Model initialized from another model. Do you want to save the initialized model?",
            ):
                filename = filedialog.asksaveasfilename(
                    defaultextension=".pth", filetypes=[("PyTorch Model", "*.pth")]
                )
                if filename:
                    self.save(filename)
                    logging.info(f"Initialized model saved to {filename}")
        except Exception as e:
            logging.error(f"Error initializing from another model: {e}")
            raise

    def mutate(self, mutation_rate: float = 0.1) -> None:
        """
        Mutate the network's weights and biases.

        Args:
            mutation_rate (float): Probability of mutation for each weight and bias.
        """
        try:
            for param in self.model.parameters():
                if torch.rand(1).item() < mutation_rate:
                    param.data += torch.randn_like(param) * mutation_rate
                    logging.debug(f"Mutated parameter with shape {param.shape}")
            logging.info("Applied mutations with mutation rate: %f", mutation_rate)
        except Exception as e:
            logging.error("Error mutating model parameters: %s", e)
            raise

    def inherit(self, other: "KANWrapper") -> None:
        """
        Inherit weights and biases from another KAN instance.

        Args:
            other (KANWrapper): Another KAN instance to inherit from.
        """
        try:
            self.model.load_state_dict(other.model.state_dict())
            logging.info("Inherited model parameters from another instance.")
        except Exception as e:
            logging.error("Error inheriting model parameters: %s", e)
            raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        try:
            output = self.model(x)
            logging.debug("Forward pass completed with input shape: %s", x.shape)
            return output
        except Exception as e:
            logging.error("Error in forward pass: %s", e)
            raise

    def train_model(
        self,
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        batch_size: int = 32,
        validation_split: float = 0.2,
        shuffle: bool = True,
    ) -> None:
        """
        Train the KAN model.

        Args:
            train_data (torch.Tensor): Training data.
            train_labels (torch.Tensor): Training labels.
            batch_size (int): Batch size for training.
            validation_split (float): Fraction of data to use for validation.
            shuffle (bool): Whether to shuffle the data before splitting.
        """
        try:
            # Split data into training and validation sets
            train_dataset = TensorDataset(train_data, train_labels)
            train_size = int((1 - validation_split) * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size]
            )
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=shuffle
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # Training loop
            for epoch in range(self.epochs):
                self.model.train()
                epoch_loss = 0.0
                for batch_data, batch_labels in train_loader:
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_data)
                    loss = torch.nn.functional.mse_loss(outputs, batch_labels)
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()

                logging.info(
                    f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss/len(train_loader)}"
                )

                # Validation loop
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for val_data, val_labels in val_loader:
                        val_outputs = self.model(val_data)
                        val_loss += torch.nn.functional.mse_loss(
                            val_outputs, val_labels
                        ).item()
                logging.info(f"Validation Loss: {val_loss/len(val_loader)}")
        except Exception as e:
            logging.error("Error during training: %s", e)
            raise


# Main block for executing script (if needed)
if __name__ == "__main__":
    # Example usage (replace with actual parameters and data as needed)
    width = [10, 20, 10]
    grid = 5
    k = 3
    learnable_params = {"lr": 0.01}
    kan_wrapper = KANWrapper(width, grid, k, learnable_params, lr=0.01, epochs=10)
    # Sample data (replace with actual data)
    train_data = torch.randn(100, 10)
    train_labels = torch.randn(100, 1)
    kan_wrapper.train_model(train_data, train_labels)
