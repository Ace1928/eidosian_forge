import os
from os.path import join
import shutil
import time
import uuid
from lion_pytorch import Lion
import pytest
import torch
import bitsandbytes as bnb
import bitsandbytes.functional as F
from tests.helpers import describe_dtype, id_formatter
@pytest.mark.parametrize('dim1', [1024], ids=id_formatter('dim1'))
@pytest.mark.parametrize('dim2', [32, 1024, 4097], ids=id_formatter('dim2'))
@pytest.mark.parametrize('gtype', [torch.float32, torch.float16], ids=describe_dtype)
def test_global_config(dim1, dim2, gtype):
    if dim1 == 1 and dim2 == 1:
        return
    p1 = torch.randn(dim1, dim2, device='cpu', dtype=gtype) * 0.1
    p2 = torch.randn(dim1, dim2, device='cpu', dtype=gtype) * 0.1
    p3 = torch.randn(dim1, dim2, device='cpu', dtype=gtype) * 0.1
    mask = torch.rand_like(p2) < 0.1
    beta1 = 0.9
    beta2 = 0.999
    lr = 0.001
    eps = 1e-08
    bnb.optim.GlobalOptimManager.get_instance().initialize()
    bnb.optim.GlobalOptimManager.get_instance().override_config(p3, 'optim_bits', 8)
    bnb.optim.GlobalOptimManager.get_instance().register_parameters([p1, p2, p3])
    p1 = p1.cuda()
    p2 = p2.cuda()
    p3 = p3.cuda()
    adam2 = bnb.optim.Adam([p1, p2, p3], lr, (beta1, beta2), eps)
    if gtype == torch.float32:
        atol, rtol = (1e-06, 1e-05)
    else:
        atol, rtol = (0.0001, 0.001)
    for i in range(50):
        g1 = torch.randn(dim1, dim2, device='cuda', dtype=gtype) * 0.1 + 0.001
        g2 = torch.randn(dim1, dim2, device='cuda', dtype=gtype) * 0.1 + 0.001
        g3 = torch.randn(dim1, dim2, device='cuda', dtype=gtype) * 0.1 + 0.001
        p1.grad = g1
        p2.grad = g2
        p3.grad = g3
        adam2.step()
        assert adam2.state[p3]['state1'].dtype == torch.uint8
        assert adam2.state[p3]['state2'].dtype == torch.uint8