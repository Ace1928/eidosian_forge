from contextlib import nullcontext
import os
from tempfile import TemporaryDirectory
import pytest
import torch
import bitsandbytes as bnb
from bitsandbytes import functional as F
from bitsandbytes.autograd import get_inverse_transform_indices, undo_layout
from bitsandbytes.nn.modules import Linear8bitLt
from tests.helpers import (
def test_linear_no_igemmlt():
    linear = torch.nn.Linear(1024, 3072)
    x = torch.randn(3, 1024, dtype=torch.half)
    linear_custom = Linear8bitLt(linear.in_features, linear.out_features, linear.bias is not None, has_fp16_weights=False, threshold=6.0)
    linear_custom.state.force_no_igemmlt = True
    linear_custom.weight = bnb.nn.Int8Params(linear.weight.data.clone(), requires_grad=False, has_fp16_weights=False).to(linear.weight.dtype)
    linear_custom.bias = linear.bias
    linear_custom = linear_custom.cuda()
    linear = linear.half().cuda()
    x_ref = x.clone().cuda().requires_grad_(True)
    x_ours = x.clone().cuda().requires_grad_(True)
    fx_ref = linear(x_ref).float()
    grad_proj = torch.randn_like(fx_ref)
    (fx_ref * grad_proj).mean().backward()
    fx_ours = linear_custom(x_ours).float()
    (fx_ours * grad_proj).mean().backward()
    assert torch.allclose(fx_ref, fx_ours, atol=0.02)
    assert torch.allclose(x_ref.grad, x_ours.grad, atol=0.01)
    assert not linear_custom.state.has_fp16_weights
    assert linear_custom.state.CB is not None
    assert linear_custom.state.CxB is None