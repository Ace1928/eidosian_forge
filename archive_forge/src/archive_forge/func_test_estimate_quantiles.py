from itertools import product
import math
import random
import time
import einops
import numpy as np
import pytest
from scipy.stats import norm
import torch
import bitsandbytes as bnb
from bitsandbytes import functional as F
from tests.helpers import (
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16], ids=['float', 'half'])
def test_estimate_quantiles(dtype):
    A = torch.rand(1024, 1024, device='cuda')
    A = A.to(dtype)
    code = F.estimate_quantiles(A)
    percs = torch.linspace(1 / 512, 511 / 512, 256, device=A.device)
    torch.testing.assert_close(percs, code, atol=0.001, rtol=0.01)
    A = torch.randn(1024, 1024, device='cuda')
    A = A.to(dtype)
    code = F.estimate_quantiles(A)
    quantiles = torch.quantile(A.float(), percs)
    diff = torch.abs(code - quantiles)
    assert (diff > 0.05).sum().item() == 0