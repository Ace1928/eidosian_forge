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
def test_kbit_quantile_estimation():
    for i in range(100):
        data = torch.randn(1024, 1024, device='cuda')
        for bits in range(2, 9):
            p = np.linspace(0.00013, 1 - 0.00013, 2 ** bits)
            val1 = torch.Tensor(norm.ppf(p)).cuda()
            val2 = F.estimate_quantiles(data, offset=0, num_quantiles=2 ** bits)
            err = torch.abs(val1 - val2).mean()
            assert err < 0.038
    for i in range(100):
        data = torch.randn(1024, 1024, device='cuda')
        for bits in range(2, 4):
            total_values = 2 ** bits - 1
            p = np.linspace(0, 1, 2 * total_values + 1)
            idx = np.arange(1, 2 * total_values + 1, 2)
            p = p[idx]
            offset = 1 / (2 * total_values)
            p = np.linspace(offset, 1 - offset, total_values)
            val1 = torch.Tensor(norm.ppf(p)).cuda()
            val2 = F.estimate_quantiles(data, num_quantiles=2 ** bits - 1)
            err = torch.abs(val1 - val2).mean()
            assert err < 0.035