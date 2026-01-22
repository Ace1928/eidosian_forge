import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Callable, List
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants
MD_DIRECTORY: str = "/home/lloyd/Downloads/gpt_chats"
PROCESSED_DATA_DIR: str = "/media/lloyd/Aurora_M2/dandata/processed_data/"
SYNTHETIC_DATA_DIR: str = "/media/lloyd/Aurora_M2/dandata/synthetic_data/"
UNTRAINED_MODEL_PATH: str = "/media/lloyd/Aurora_M2/dandata/untrained_dan_model.pkl"
REGRESSION_TRAINED_MODEL_PATH: str = (
    "/media/lloyd/Aurora_M2/dandata/regression_trained_dan_model.pkl"
)
TEXT_TRAINED_MODEL_PATH: str = (
    "/media/lloyd/Aurora_M2/dandata/text_trained_dan_model.pkl"
)
INPUT_SIZE: int = 512
OUTPUT_SIZE: int = 56
NUM_EPOCHS: int = 50
SYNTHETIC_DATA_SIZE: int = 1000000
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42
LR: float = 0.001
MAX_DATASET_SIZE_GB: int = 10
BATCH_SIZE: int = 100


def get_dataset_size_gb(dataset: np.ndarray) -> float:
    """
    Calculate the size of the dataset in gigabytes.

    Parameters:
    dataset (np.ndarray): The dataset to measure.

    Returns:
    float: Size of the dataset in gigabytes.
    """
    return dataset.nbytes / (1024**3)


def create_new_synthetic_dataset() -> tuple[np.ndarray, np.ndarray]:
    """
    Create a new synthetic dataset for regression.

    Returns:
    tuple: Features and labels of the synthetic dataset.
    """
    X, y = make_regression(
        n_samples=SYNTHETIC_DATA_SIZE,
        n_features=INPUT_SIZE,
        n_targets=OUTPUT_SIZE,
        noise=0.1,
    )
    y = y.reshape(-1, 1)
    return X, y


def save_dataset(X: np.ndarray, y: np.ndarray, dataset_dir: str, dataset_name: str):
    """
    Save the dataset to the specified directory.

    Parameters:
    X (np.ndarray): Features.
    y (np.ndarray): Labels.
    dataset_dir (str): Directory to save the dataset.
    dataset_name (str): Name of the dataset file.
    """
    os.makedirs(dataset_dir, exist_ok=True)
    np.save(os.path.join(dataset_dir, f"{dataset_name}_data.npy"), X)
    np.save(os.path.join(dataset_dir, f"{dataset_name}_labels.npy"), y)
    logger.info(f"Saved dataset {dataset_name} to {dataset_dir}")


def load_dataset(dataset_dir: str, dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the dataset from the specified directory.

    Parameters:
    dataset_dir (str): Directory to load the dataset from.
    dataset_name (str): Name of the dataset file.

    Returns:
    tuple: Features and labels.
    """
    X = np.load(os.path.join(dataset_dir, f"{dataset_name}_data.npy"))
    y = np.load(os.path.join(dataset_dir, f"{dataset_name}_labels.npy"))
    logger.info(f"Loaded dataset {dataset_name} from {dataset_dir}")
    return X, y


def train_on_datasets(
    model: nn.Module,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    dataset_dir: str,
    dataset_prefix: str,
):
    """
    Train the model on datasets stored in the specified directory.

    Parameters:
    model (nn.Module): The neural network model.
    criterion (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): The loss function.
    optimizer (torch.optim.Optimizer): The optimizer.
    num_epochs (int): Number of epochs to train the model.
    dataset_dir (str): Directory containing the datasets.
    dataset_prefix (str): Prefix of the dataset files.
    """
    dataset_index = 0
    while True:
        dataset_name = f"{dataset_prefix}_{dataset_index}"
        if not os.path.exists(os.path.join(dataset_dir, f"{dataset_name}_data.npy")):
            break
        X, y = load_dataset(dataset_dir, dataset_name)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        history = train_model(
            model, criterion, optimizer, train_loader, val_loader, num_epochs
        )
        torch.save(model, REGRESSION_TRAINED_MODEL_PATH)
        logger.info(f"Saved the model after training on dataset {dataset_name}.")
        plot_loss(history)
        dataset_index += 1


def main():
    """
    Main function to orchestrate the training process.
    """
    try:
        os.makedirs(os.path.dirname(UNTRAINED_MODEL_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(REGRESSION_TRAINED_MODEL_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(TEXT_TRAINED_MODEL_PATH), exist_ok=True)
        os.makedirs(SYNTHETIC_DATA_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

        if os.path.exists(UNTRAINED_MODEL_PATH):
            dan = torch.load(UNTRAINED_MODEL_PATH)
            logger.info("Loaded the most recent untrained model.")
        else:
            dan = DynamicActivationNeuron(INPUT_SIZE, OUTPUT_SIZE, SCALE_OUTPUT)
            torch.save(dan, UNTRAINED_MODEL_PATH)
            logger.info("Created and saved a new untrained model.")

        dataset_index = 0
        while True:
            X, y = create_new_synthetic_dataset()
            if get_dataset_size_gb(X) > MAX_DATASET_SIZE_GB:
                break
            save_dataset(X, y, SYNTHETIC_DATA_DIR, f"synthetic_{dataset_index}")
            dataset_index += 1

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(dan.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        train_on_datasets(
            dan, criterion, optimizer, NUM_EPOCHS, SYNTHETIC_DATA_DIR, "synthetic"
        )

        train_loader, val_loader, INPUT_SIZE, OUTPUT_SIZE = load_and_preprocess_data(
            MD_DIRECTORY, VECTORIZER, INPUT_SIZE, OUTPUT_SIZE
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(dan.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        history = train_model(
            dan, criterion, optimizer, train_loader, val_loader, NUM_EPOCHS
        )
        torch.save(dan, TEXT_TRAINED_MODEL_PATH)
        logger.info("Saved the text trained model.")
        plot_loss(history)

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise
