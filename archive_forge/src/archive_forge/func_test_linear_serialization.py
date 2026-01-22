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
@pytest.mark.parametrize('has_fp16_weights', TRUE_FALSE, ids=id_formatter('has_fp16_weights'))
@pytest.mark.parametrize('serialize_before_forward', TRUE_FALSE, ids=id_formatter('serialize_before_forward'))
@pytest.mark.parametrize('deserialize_before_cuda', TRUE_FALSE, ids=id_formatter('deserialize_before_cuda'))
@pytest.mark.parametrize('force_no_igemmlt', TRUE_FALSE, ids=id_formatter('force_no_igemmlt'))
@pytest.mark.parametrize('save_before_forward', TRUE_FALSE, ids=id_formatter('save_before_forward'))
@pytest.mark.parametrize('load_before_cuda', TRUE_FALSE, ids=id_formatter('load_before_cuda'))
def test_linear_serialization(has_fp16_weights, serialize_before_forward, deserialize_before_cuda, force_no_igemmlt, save_before_forward, load_before_cuda):
    linear = torch.nn.Linear(32, 96)
    x = torch.randn(3, 32, dtype=torch.half)
    linear_custom = Linear8bitLt(linear.in_features, linear.out_features, linear.bias is not None, has_fp16_weights=has_fp16_weights, threshold=6.0)
    if force_no_igemmlt:
        linear_custom.state.force_no_igemmlt = True
    linear_custom.weight = bnb.nn.Int8Params(linear.weight.data.clone(), requires_grad=has_fp16_weights, has_fp16_weights=has_fp16_weights)
    linear_custom.bias = linear.bias
    linear_custom = linear_custom.cuda()
    if serialize_before_forward:
        state_dict_8bit = linear_custom.state_dict()
    if save_before_forward:
        bytes_8bit = torch_save_to_buffer(linear_custom)
    x_first = x.clone().cuda().requires_grad_(True)
    fx_first = linear_custom(x_first).float()
    grad_proj = torch.randn_like(fx_first)
    (fx_first * grad_proj).mean().backward()
    if not serialize_before_forward:
        state_dict_8bit = linear_custom.state_dict()
    if not save_before_forward:
        bytes_8bit = torch_save_to_buffer(linear_custom)
    with TemporaryDirectory() as tmpdir:
        state_path_8bit = os.path.join(tmpdir, 'state_8bit.pth')
        state_path = os.path.join(tmpdir, 'state.pth')
        torch.save(linear.state_dict(), state_path)
        torch.save(state_dict_8bit, state_path_8bit)
        if not has_fp16_weights:
            assert os.path.getsize(state_path_8bit) < 0.5 * os.path.getsize(state_path)
        new_state_dict = torch.load(state_path_8bit)
    new_linear_custom = Linear8bitLt(linear.in_features, linear.out_features, linear.bias is not None, has_fp16_weights=has_fp16_weights, threshold=6.0)
    if force_no_igemmlt:
        new_linear_custom.state.force_no_igemmlt = True
    if deserialize_before_cuda:
        with nullcontext() if has_fp16_weights else pytest.raises(RuntimeError):
            new_linear_custom.load_state_dict(new_state_dict, strict=True)
    if load_before_cuda:
        new_linear_custom2 = torch_load_from_buffer(bytes_8bit)
    new_linear_custom = new_linear_custom.cuda()
    if not deserialize_before_cuda:
        new_linear_custom.load_state_dict(new_state_dict, strict=True)
    if not load_before_cuda:
        new_linear_custom2 = torch_load_from_buffer(bytes_8bit)
    x_second = x.clone().cuda().requires_grad_(True)
    fx_second = new_linear_custom(x_second).float()
    (fx_second * grad_proj).mean().backward()
    x_third = x.clone().cuda().requires_grad_(True)
    fx_third = new_linear_custom2(x_third).float()
    (fx_third * grad_proj).mean().backward()
    if has_fp16_weights or not deserialize_before_cuda:
        assert torch.allclose(fx_first, fx_second, atol=1e-05)
        assert torch.allclose(x_first.grad, x_second.grad, atol=1e-05)
    assert torch.allclose(fx_first, fx_third, atol=1e-05)
    assert torch.allclose(x_first.grad, x_third.grad, atol=1e-05)