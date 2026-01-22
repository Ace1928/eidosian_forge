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
def test_zeropoint():

    def quant_zp(x):
        dtype = x.dtype
        x = x.float()
        dyna = x.max() - x.min()
        if dyna == 0:
            dyna = 1
        qx = 254.0 / dyna
        minx = x.min()
        zpx = torch.round(x.min() * qx) - 127
        x = qx * x + zpx
        return (x, qx, zpx)
    batch = 2
    seq = 512
    model = 1024
    hidden = 4 * model
    A = torch.randn(batch * seq, model, device='cuda').half() * 0.1
    B = torch.randn(model, hidden, device='cuda').half() * 0.1
    C0 = torch.matmul(A, B)
    A = A.float()
    B = B.float()
    C1 = torch.matmul(A, B)
    C3 = bnb.matmul(A.half(), B.t().contiguous().half())
    zp = 1
    C2 = torch.matmul(A, B - zp)
    C2 -= A.sum(1).view(-1, 1) * zp
    ca, cqa, cza = quant_zp(A)
    zp = 1
    scale = 2.0
    C5 = torch.matmul(A * scale - zp, B)
    C5 += B.sum(0) * zp
    C5 /= scale
    CA, qa, zpa = quant_zp(A)
    C4 = torch.matmul(CA, B)
    C4 -= B.sum(0) * zpa
    C4 /= qa
    zpb = 1
    zpa = 1
    qa = 2
    qb = 2
    C6 = torch.matmul(A * qa + zpa, B * qb + zpb)
    C6 -= qb * B.sum(0).view(1, -1) * zpa + qa * A.sum(1).view(-1, 1) * zpb
    C6 -= zpa * zpb * A.shape[1]
    C6 /= qa * qb
    CA, qa, zpa = quant_zp(A)
    CB, qb, zpb = quant_zp(B)
    C7 = torch.matmul(CA, CB)
    C7 -= qb * B.sum(0).view(1, -1) * zpa + qa * A.sum(1).view(-1, 1) * zpb
    C7 -= zpa * zpb * A.shape[1]
    C7 /= qa * qb
    err1 = torch.abs(C1 - C2).mean().item()
    err2 = torch.abs(C1 - C3).mean().item()
    err3 = torch.abs(C1 - C4).mean().item()
    err4 = torch.abs(C1 - C5).mean().item()
    err5 = torch.abs(C1 - C6).mean().item()
    err6 = torch.abs(C1 - C7).mean().item()
    print(err1, err2, err3, err4, err5, err6)