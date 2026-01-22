import math
import einops
import pytest
import torch
from torch import nn
import bitsandbytes as bnb
from tests.helpers import id_formatter
@pytest.mark.parametrize('module', module_dict.values(), ids=module_dict.keys())
def test_kbit_backprop(module):
    b = 17
    dim1 = 37
    dim2 = 83
    ref = nn.Sequential(*[torch.nn.Linear(dim1, dim2), torch.nn.Linear(dim2, 10)])
    ref[1].weight.requires_grad = False
    torch.nn.init.kaiming_normal_(ref[0].weight)
    torch.nn.init.kaiming_normal_(ref[1].weight)
    kbit = nn.Sequential(*[torch.nn.Linear(dim1, dim2), module(dim2, 10)])
    kbit[0].weight.detach().copy_(ref[0].weight)
    kbit[1].weight.detach().copy_(ref[1].weight)
    kbit[0].bias.detach().copy_(ref[0].bias)
    kbit[1].bias.detach().copy_(ref[1].bias)
    ref = ref.half().cuda()
    kbit = kbit.half().cuda()
    kbit = kbit.half().to('cuda')
    errs1 = []
    errs2 = []
    relerrs1 = []
    relerrs2 = []
    for i in range(100):
        batch = torch.randn(b, dim1).half().cuda()
        out1 = ref(batch)
        out2 = kbit(batch)
        out1.mean().backward()
        out2.mean().backward()
        grad1 = ref[0].weight.grad
        grad2 = kbit[0].weight.grad
        bgrad1 = ref[0].bias.grad
        bgrad2 = kbit[0].bias.grad
        err1 = (out1 - out2).abs().float()
        err2 = (grad1 - grad2).abs().float()
        relerr1 = err1 / (out1.abs().float() + 1e-09)
        relerr2 = err2 / (grad1.abs().float() + 1e-09)
        errs1.append(err1.mean().item())
        errs2.append(err2.mean().item())
        relerrs1.append(relerr1.mean().item())
        relerrs2.append(relerr2.mean().item())
        if isinstance(module, bnb.nn.Linear8bitLt):
            assert_all_approx_close(grad1, grad2, atol=0.008, rtol=0.05, count=1)
            torch.testing.assert_close(bgrad1, bgrad2, atol=0.008, rtol=0.05)
        else:
            assert_all_approx_close(grad1, grad2, atol=0.015, rtol=0.05, count=1)
            torch.testing.assert_close(bgrad1, bgrad2, atol=0.02, rtol=0.05)
        ref.zero_grad()
        kbit.zero_grad()
        assert kbit[0].weight.grad is None or kbit[0].weight.grad.sum().item() == 0
        assert kbit[0].weight.grad is None or kbit[0].bias.grad.sum().item() == 0