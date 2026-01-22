import math
import einops
import pytest
import torch
from torch import nn
import bitsandbytes as bnb
from tests.helpers import id_formatter
@pytest.mark.parametrize('module', [lambda n_in, n_out, bias=True: bnb.nn.Linear8bitLt(n_in, n_out, bias=bias, has_fp16_weights=False), bnb.nn.LinearFP4], ids=['Int8Lt', 'FP4'])
def test_linear_kbit_fp32_bias(module):
    l1 = module(32, 64).cuda()
    assert l1.weight.dtype in [torch.int8, torch.uint8]
    assert l1.bias.dtype == torch.float32
    for i in range(100):
        b1 = torch.randn(16, 8, 32, device='cuda').half()
        o1 = l1(b1)
        assert l1.bias.dtype == torch.float16
    l1 = module(32, 64, bias=False).cuda()
    assert l1.weight.dtype in [torch.int8, torch.uint8]
    assert l1.bias is None
    for i in range(100):
        b1 = torch.randn(16, 8, 32, device='cuda').half()
        o1 = l1(b1)
        assert l1.bias is None