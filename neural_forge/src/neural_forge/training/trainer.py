import torch
import torch.nn as torch_nn
import torch.optim as optim
from neural_forge.models.integrated import EidosianSingularityModel


class SingularityTrainer:
    """
    Simplified GaLore + Sophia inspired trainer for the Eidosian Neural Forge.
    Optimized for high-speed CPU training on edge devices.
    """
    def __init__(self, model: EidosianSingularityModel, lr: float = 1e-4):
        self.model = model
        # For the prototype, we use AdamW. 
        # Full GaLore/Sophia will be implemented in Phase 4.
        self.optimizer = optim.AdamW(model.parameters(), lr=lr)
        self.criterion = torch_nn.CrossEntropyLoss()

    def train_step(self, x: torch.Tensor, y: torch.Tensor):
        self.optimizer.zero_grad()
        output = self.model(x)

        # Flatten for loss
        loss = self.criterion(output.view(-1, output.size(-1)), y.view(-1))
        loss.backward()

        # Gradient Clipping for BitNet stability
        torch_nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()
        return loss.item()
