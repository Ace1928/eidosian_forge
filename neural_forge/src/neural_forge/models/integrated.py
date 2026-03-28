from __future__ import annotations

import torch
import torch.nn as torch_nn
from typing import Optional

from ..core.modality import ModalityClassifier, SharedLatentSpace, Modality
from .singularity import NeuralForgeSingularity

class CognitiveConflictMonitor(torch_nn.Module):
    """
    Monitors expert divergence to detect novelty and trigger Hypernetwork synthesis.
    """
    def __init__(self, threshold: float = 0.7):
        super().__init__()
        self.threshold = threshold

    def forward(self, expert_outputs: list[torch.Tensor]) -> torch.Tensor:
        # expert_outputs is a list of [B, L, D]
        if len(expert_outputs) < 2:
            return torch.zeros(expert_outputs[0].shape[0], device=expert_outputs[0].device)
            
        stack = torch.stack(expert_outputs) # [num_experts, B, L, D]
        # Calculate variance across experts as a proxy for conflict/novelty
        variance = torch.var(stack, dim=0).mean(dim=[1, 2]) # [B]
        return variance

class EidosianSingularityModel(torch_nn.Module):
    """
    The complete, integrated Eidosian Neural Forge model.
    0.5B Target, Omni-Modal, MoE + Hypernetwork + Cognitive Conflict.
    """
    def __init__(self, vocab_size: int = 256, dim: int = 768, depth: int = 12):
        super().__init__()
        self.dim = dim
        
        # 1. Modality Control
        self.classifier = ModalityClassifier(dim)
        self.latent_space = SharedLatentSpace(dim)
        
        # 2. Base Backbone
        self.backbone = NeuralForgeSingularity(vocab_size, dim, depth)
        
        # 3. Novelty Detection
        self.conflict_monitor = CognitiveConflictMonitor()

    def forward(self, x: torch.Tensor, modality: Optional[Modality] = None) -> torch.Tensor:
        # Token IDs are treated as the native text path and bypass modality projection.
        if modality is None and x.ndim == 2 and x.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            modality = Modality.TEXT
        if modality == Modality.TEXT and x.ndim == 2 and x.dtype in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            return self.backbone(x)

        if x.ndim != 3:
            raise ValueError("Non-token inputs must be shaped [batch, length, dim].")

        # 1. Automatic Modality Detection is only safe once inputs are already in shared-dim feature space.
        if modality is None:
            if x.shape[-1] != self.dim:
                raise ValueError("Explicit modality is required for non-text features outside the shared latent width.")
            logits = self.classifier(x)
            predicted = torch.argmax(logits, dim=-1)
            mod_idx = torch.bincount(predicted, minlength=len(Modality)).argmax().item()
            modality = Modality(mod_idx)

        # 2. Project into Shared Latent Space
        x = self.latent_space(x, modality)

        # 3. Process through Backbone
        return self.backbone.forward_embedded(x)
