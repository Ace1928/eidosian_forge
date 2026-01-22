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
@pytest.mark.parametrize('dim1', get_test_dims(1, 4 * 1024, n=2), ids=id_formatter('dim1'))
@pytest.mark.parametrize('dim2', get_test_dims(1, 4 * 1024, n=2), ids=id_formatter('dim2'))
def test_coo_double_quant(dim1, dim2):
    threshold = 3.0
    for i in range(k):
        A = torch.randn(dim1, dim2, device='cuda').half()
        idx = torch.abs(A) >= threshold
        CA2, CAt, statsA, statsAt, coo_tensor = F.double_quant(A)
        CA, CAt, statsA, statsAt, coo_tensor = F.double_quant(A, threshold=threshold)
        if coo_tensor is not None:
            A1 = A * idx
            A2 = torch.zeros_like(A)
            A2[coo_tensor.rowidx.long(), coo_tensor.colidx.long()] = coo_tensor.values
            torch.testing.assert_close(A1, A2)
            A1 = A * (idx == 0)
            A2 = (CA.float() * statsA.unsqueeze(1) / 127).half()
            torch.testing.assert_close(A * (idx == 0), A2, rtol=0.05, atol=0.015)