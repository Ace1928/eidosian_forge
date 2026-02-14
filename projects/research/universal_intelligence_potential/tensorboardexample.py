import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard.writer import SummaryWriter
import time
import logging
from typing import Tuple, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to generate synthetic data
def generate_data(num_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
    X = torch.randn(num_samples, 10)
    y = torch.sum(X, dim=1, keepdim=True) + torch.randn(num_samples, 1) * 0.1
    return X, y

# Function to train the model
def train_model(device: str, dataloader: DataLoader, epochs: int = 5) -> float:
    try:
        device = torch.device(device)
        model = SimpleNN().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        writer = SummaryWriter(f'runs/exp_{device}')
        total_time = 0

        for epoch in range(epochs):
            start_time = time.time()
            for i, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                if i % 100 == 0:
                    writer.add_scalar('training_loss', loss.item(), epoch * len(dataloader) + i)

            epoch_time = time.time() - start_time
            total_time += epoch_time
            logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Time: {epoch_time:.2f} seconds")

        writer.close()
        return total_time
    except Exception as e:
        logging.error(f"An error occurred during training: {e}")
        return float('inf')

# Main function to execute the training
def main():
    X, y = generate_data()
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Train on CPU
    cpu_time = train_model('cpu', dataloader)

    # Train on GPU
    cuda_time = None
    if torch.cuda.is_available():
        cuda_time = train_model('cuda', dataloader)

    logging.info(f"Total training time on CPU: {cpu_time:.2f} seconds")
    if cuda_time is not None:
        logging.info(f"Total training time on GPU: {cuda_time:.2f} seconds")

if __name__ == "__main__":
    main()
