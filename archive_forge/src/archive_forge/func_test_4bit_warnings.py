import math
import einops
import pytest
import torch
from torch import nn
import bitsandbytes as bnb
from tests.helpers import id_formatter
def test_4bit_warnings():
    dim1 = 64
    with pytest.warns(UserWarning, match='inference or training'):
        net = nn.Sequential(*[bnb.nn.Linear4bit(dim1, dim1, compute_dtype=torch.float32) for i in range(10)])
        net = net.cuda()
        inp = torch.rand(10, dim1).cuda().half()
        net(inp)
    with pytest.warns(UserWarning, match='inference.'):
        net = nn.Sequential(*[bnb.nn.Linear4bit(dim1, dim1, compute_dtype=torch.float32) for i in range(10)])
        net = net.cuda()
        inp = torch.rand(1, dim1).cuda().half()
        net(inp)
    with pytest.warns(UserWarning) as record:
        net = nn.Sequential(*[bnb.nn.Linear4bit(dim1, dim1, compute_dtype=torch.float32) for i in range(10)])
        net = net.cuda()
        inp = torch.rand(10, dim1).cuda().half()
        net(inp)
        net = nn.Sequential(*[bnb.nn.Linear4bit(dim1, dim1, compute_dtype=torch.float32) for i in range(10)])
        net = net.cuda()
        inp = torch.rand(1, dim1).cuda().half()
        net(inp)
    assert len(record) == 2