from __future__ import annotations

import torch
import torch.nn as torch_nn
from ..core.layers import BitLinear, BitRMSNorm

class BitAttention(torch_nn.Module):
    """Multi-head attention implemented with BitLinear layers."""
    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        
        self.qkv = BitLinear(dim, dim * 3, bias=False)
        self.proj = BitLinear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.proj(x)

class BitBlock(torch_nn.Module):
    """A single Transformer block using BitLinear and BitRMSNorm."""
    def __init__(self, dim: int, ff_dim: int):
        super().__init__()
        self.attn_norm = BitRMSNorm(dim)
        self.attn = BitAttention(dim)
        
        self.ff_norm = BitRMSNorm(dim)
        self.ff = torch_nn.Sequential(
            BitLinear(dim, ff_dim, bias=False),
            torch_nn.GELU(),
            BitLinear(ff_dim, dim, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ff(self.ff_norm(x))
        return x

class BitTransformer(torch_nn.Module):
    """The base backbone for the 0.5B Eidosian Neural Forge model."""
    def __init__(self, vocab_size: int, dim: int = 1024, depth: int = 12, ff_dim: int = 2816):
        super().__init__()
        self.embed = torch_nn.Embedding(vocab_size, dim)
        self.blocks = torch_nn.ModuleList([BitBlock(dim, ff_dim) for _ in range(depth)])
        self.final_norm = BitRMSNorm(dim)
        self.head = BitLinear(dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.head(self.final_norm(x))
