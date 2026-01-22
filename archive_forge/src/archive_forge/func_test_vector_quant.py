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
@pytest.mark.parametrize('dim1', get_test_dims(1, 64, n=1), ids=id_formatter('dim1'))
@pytest.mark.parametrize('dim2', get_test_dims(32, 128, n=1), ids=id_formatter('dim2'))
@pytest.mark.parametrize('dim3', get_test_dims(32, 256, n=1), ids=id_formatter('dim3'))
def test_vector_quant(dim1, dim2, dim3):
    dim2 = dim2 - dim2 % 16
    dim3 = dim3 - dim3 % 16
    for i in range(k):
        A = torch.randn(size=(dim2, dim3), device='cuda')
        qA, SA = F.vectorwise_quant(A, dim=0)
        A1 = F.vectorwise_dequant(qA, SA)
        n = A1.numel()
        assert_all_approx_close(A1, A, atol=0.01, rtol=0.1, count=int(n * 0.002))