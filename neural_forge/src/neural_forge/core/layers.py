from __future__ import annotations

import torch
import torch.nn as torch_nn
import torch.nn.functional as F

def activation_quant(x: torch.Tensor) -> torch.Tensor:
    """Quantize activations to 8-bit."""
    scale = 127.0 / x.abs().max().clamp(min=1e-5)
    y = (x * scale).round().clamp(-128, 127)
    return y / scale

def weight_quant(w: torch.Tensor) -> torch.Tensor:
    """Quantize weights to 1.58-bit (ternary: -1, 0, 1)."""
    scale = w.abs().mean().clamp(min=1e-5)
    w_quant = (w / scale).round().clamp(-1, 1)
    return w_quant, scale

class BitLinear(torch_nn.Linear):
    """
    BitLinear Layer as proposed in BitNet b1.58.
    Replaces standard Linear layer with 1.58-bit weights and 8-bit activations.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Quantize Activations
        x_quant = activation_quant(x)
        
        # 2. Quantize Weights (using Straight-Through Estimator for training)
        w = self.weight
        w_quant, scale = weight_quant(w)
        
        # Apply STE: Use quantized weights in forward, but gradients flow to original weights
        w_final = w + (w_quant * scale - w).detach()
        
        # 3. Perform Linear Operation (effectively MatMul-free in optimized C++ runtimes)
        return F.linear(x_quant, w_final, self.bias)

class BitRMSNorm(torch_nn.Module):
    """Root Mean Square Layer Normalization for BitNet stability."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch_nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm_x * self.weight
