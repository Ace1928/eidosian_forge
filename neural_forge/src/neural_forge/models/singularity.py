from __future__ import annotations

import torch
import torch.nn as torch_nn
from ..core.layers import BitRMSNorm, BitLinear
from ..core.moe import BitMoE

class SingularityBlock(torch_nn.Module):
    """
    A single block of the Singularity model, integrating BitAttention and BitMoE.
    """
    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        # Simplified Attention for the block (standard BitAttention from backbone)
        from .backbone import BitAttention
        
        self.attn_norm = BitRMSNorm(dim)
        self.attn = BitAttention(dim, heads=heads)
        
        self.moe_norm = BitRMSNorm(dim)
        self.moe = BitMoE(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.moe(self.moe_norm(x))
        return x

class NeuralForgeSingularity(torch_nn.Module):
    """
    The Eidosian Neural Forge Singularity Model (0.5B Target).
    Universal-modal, MoE/Hypernetwork backbone.
    """
    def __init__(self, vocab_size: int, dim: int = 768, depth: int = 12):
        super().__init__()
        self.embed = torch_nn.Embedding(vocab_size, dim)
        self.blocks = torch_nn.ModuleList([SingularityBlock(dim) for _ in range(depth)])
        self.final_norm = BitRMSNorm(dim)
        self.head = BitLinear(dim, vocab_size, bias=False)

    def forward_embedded(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.head(self.final_norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Token-ID path for text. Pre-embedded multimodal inputs should use forward_embedded().
        return self.forward_embedded(self.embed(x))
