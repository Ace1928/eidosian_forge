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
@pytest.mark.parametrize('dim1', [32], ids=id_formatter('dim1'))
@pytest.mark.parametrize('dim2', [32], ids=id_formatter('dim2'))
@pytest.mark.parametrize('dim3', [32], ids=id_formatter('dim3'))
@pytest.mark.parametrize('dim4', [32], ids=id_formatter('dim4'))
@pytest.mark.parametrize('dims', (2,), ids=id_formatter('dims'))
def test_igemmlt_half(dim1, dim2, dim3, dim4, dims):
    formatB = F.get_special_format_str()
    for i in range(k):
        if dims == 2:
            A = torch.normal(0, 0.5, size=(dim1, dim3), device='cuda').half()
        elif dims == 3:
            A = torch.normal(0, 0.5, size=(dim1, dim2, dim3), device='cuda').half()
        B = torch.randn((dim4, dim3), device='cuda').half()
        torch.nn.init.xavier_uniform_(B)
        C1 = torch.matmul(A, B.t())
        C2 = bnb.matmul(A, B.t())
        A = A.view(-1, A.shape[-1])
        CA, CAt, statsA, statsAt, coo_tensor = F.double_quant(A)
        CB, CBt, statsB, statsBt, coo_tensor = F.double_quant(B)
        C32A, SA = F.transform(CA, 'col32')
        CxB, SB = F.transform(CB, to_order=formatB)
        out1_32, Sout1_32 = F.igemmlt(C32A, CxB, SA, SB)
        output = F.mm_dequant(out1_32, Sout1_32, statsAt, statsBt)