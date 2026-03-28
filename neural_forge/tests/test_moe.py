import torch
import pytest
from neural_forge.core.moe import BitMoE

def test_moe_shapes():
    """Verify the Mixture of Experts layer preserves output shapes."""
    dim = 128
    moe = BitMoE(dim, num_experts=4, top_k=2)
    x = torch.randn(2, 10, dim)
    out = moe(x)
    assert out.shape == (2, 10, dim)

def test_moe_routing():
    """Ensure MoE router generates valid selection scores."""
    dim = 64
    moe = BitMoE(dim, num_experts=2, top_k=1)
    x = torch.randn(1, 5, dim)
    logits = moe.router(x.mean(dim=1))
    assert logits.shape == (1, 3) # 2 static + 1 hyper
