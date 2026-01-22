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
@pytest.mark.parametrize('dim1', get_test_dims(1, 64, n=2), ids=id_formatter('dim1'))
@pytest.mark.parametrize('dim2', get_test_dims(32, 128, n=2), ids=id_formatter('dim2'))
@pytest.mark.parametrize('dim3', get_test_dims(32, 256, n=2), ids=id_formatter('dim3'))
@pytest.mark.parametrize('dim4', get_test_dims(32, 256, n=2), ids=id_formatter('dim4'))
@pytest.mark.parametrize('transpose', BOOLEAN_TUPLES, ids=id_formatter('transpose'))
def test_ibmm(dim1, dim2, dim3, dim4, transpose):
    dim2 = dim2 - dim2 % 16
    dim3 = dim3 - dim3 % 16
    dim4 = dim4 - dim4 % 16
    for i in range(k):
        shapeA = (dim1, dim3, dim2) if transpose[0] else (dim1, dim2, dim3)
        shapeB = (dim1, dim4, dim3) if transpose[1] else (dim1, dim3, dim4)
        A = torch.randint(-128, 127, size=shapeA, device='cuda').to(torch.int8)
        B = torch.randint(-128, 127, size=shapeB, device='cuda').to(torch.int8)
        if not transpose[0] and (not transpose[1]):
            out2 = torch.bmm(A.float(), B.float())
            out = F.igemm(A, B)
        elif not transpose[0] and transpose[1]:
            out2 = torch.bmm(A.float(), B.permute([0, 2, 1]).float())
            out = F.igemm(A, B.permute([0, 2, 1]))
        elif transpose[0] and (not transpose[1]):
            out2 = torch.bmm(A.permute([0, 2, 1]).float(), B.float())
            out = F.igemm(A.permute([0, 2, 1]), B)
        elif transpose[0] and transpose[1]:
            out2 = torch.bmm(A.permute([0, 2, 1]).float(), B.permute([0, 2, 1]).float())
            out = F.igemm(A.permute([0, 2, 1]), B.permute([0, 2, 1]))
        torch.testing.assert_close(out.float(), out2.float())