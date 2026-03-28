from __future__ import annotations

from enum import Enum
import torch
import torch.nn as torch_nn

class Modality(Enum):
    TEXT = 0
    IMAGE = 1
    AUDIO = 2

class ModalityClassifier(torch_nn.Module):
    """
    A lightweight supervisor that identifies the modality of the input signal.
    """
    def __init__(self, input_dim: int, num_modalities: int = 3):
        super().__init__()
        self.net = torch_nn.Sequential(
            torch_nn.Linear(input_dim, 128),
            torch_nn.ReLU(),
            torch_nn.Linear(128, num_modalities)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D) -> mean(L) -> (B, D)
        return self.net(x.mean(dim=1))

class SharedLatentSpace(torch_nn.Module):
    """
    Unified 768-dim embedding hub for all modalities.
    """
    def __init__(self, shared_dim: int = 768):
        super().__init__()
        self.shared_dim = shared_dim
        
        # Modality-specific projection layers
        self.text_proj = torch_nn.Linear(shared_dim, shared_dim)
        self.image_proj = torch_nn.Linear(512, shared_dim) # Assuming 512 for ViT features
        self.audio_proj = torch_nn.Linear(256, shared_dim) # Assuming 256 for AudioMAE

    def forward(self, x: torch.Tensor, modality: Modality) -> torch.Tensor:
        if modality == Modality.TEXT:
            return self.text_proj(x)
        elif modality == Modality.IMAGE:
            return self.image_proj(x)
        elif modality == Modality.AUDIO:
            return self.audio_proj(x)
        else:
            raise ValueError(f"Unknown modality: {modality}")
