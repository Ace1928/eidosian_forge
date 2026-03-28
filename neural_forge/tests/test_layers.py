import torch
import pytest
from neural_forge.core.layers import BitLinear, activation_quant, weight_quant

def test_weight_quantization():
    """Verify that weights are correctly quantized to ternary values."""
    w = torch.tensor([[-2.0, -0.1, 0.0, 0.1, 2.0]])
    w_quant, scale = weight_quant(w)
    # ternary values: -1, 0, 1
    assert torch.all(torch.abs(w_quant) <= 1.0)
    assert torch.all(w_quant.round() == w_quant)
    assert scale > 0

def test_bitlinear_shapes():
    """Ensure BitLinear layer preserves expected output shapes."""
    layer = BitLinear(128, 256)
    x = torch.randn(1, 10, 128)
    out = layer(x)
    assert out.shape == (1, 10, 256)

def test_straight_through_estimator():
    """Verify that gradients flow through BitLinear to the original weights."""
    layer = BitLinear(10, 10)
    x = torch.randn(1, 10)
    out = layer(x).sum()
    out.backward()
    
    assert layer.weight.grad is not None
    assert torch.any(layer.weight.grad != 0)
