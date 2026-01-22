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
@pytest.mark.parametrize('dim1', get_test_dims(1, 1 * 1024, n=2), ids=id_formatter('dim1'))
@pytest.mark.parametrize('dim2', get_test_dims(1, 1 * 1024, n=2), ids=id_formatter('dim2'))
@pytest.mark.parametrize('transposed_B', TRUE_FALSE, ids=id_formatter('transposed_B'))
def test_spmm_coo(dim1, dim2, transposed_B):
    threshold = 1.5
    dim3 = torch.randint(32, 128, size=(1,)).item()
    for i in range(k):
        A = torch.randn(dim1, dim2).cuda().half()
        if transposed_B:
            B = torch.randn(dim3, dim2).cuda().half()
        else:
            B = torch.randn(dim2, dim3).cuda().half()
        idx = torch.abs(A) >= threshold
        nnz = (idx == 1).sum().item()
        rows, cols = torch.where(idx)
        values = A[idx]
        cooA = F.COOSparseTensor(A.shape[0], A.shape[1], nnz, rows.int(), cols.int(), values)
        A2 = A * idx
        if transposed_B:
            out2 = F.spmm_coo(cooA, B.t())
            out1 = torch.matmul(A2, B.t())
        else:
            out2 = F.spmm_coo(cooA, B)
            out1 = torch.matmul(A2, B)
        assert_all_approx_close(out1, out2, rtol=0.01, atol=0.03, count=30)