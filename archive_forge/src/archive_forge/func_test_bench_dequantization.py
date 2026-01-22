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
@pytest.mark.benchmark
def test_bench_dequantization():
    a = torch.rand(1024, 1024, device='cuda').half()
    code = F.create_fp8_map(True, 3, 0, 4).cuda()
    qa, SA = F.quantize_blockwise(a, code=code)
    print(qa.max())
    max_theoretical_mu = 1024 * 1024 * 2 / 1024 ** 3 / 672 * 1000 * 1000
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(100):
        qa, SA = F.quantize_blockwise(a)
    torch.cuda.synchronize()