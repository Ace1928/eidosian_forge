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
@pytest.mark.parametrize('double_quant', TRUE_FALSE, ids=lambda double_quant: f'DQ_{double_quant}')
@pytest.mark.parametrize('storage_type', ['nf4', 'fp4'])
@pytest.mark.parametrize('kind', ['fc1', 'fc2', 'attn', 'attn_packed'])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32], ids=describe_dtype)
@pytest.mark.parametrize('quant_storage', [torch.uint8, torch.float16, torch.bfloat16, torch.float32], ids=describe_dtype)
def test_gemv_4bit(dtype, storage_type, quant_storage, double_quant, kind):
    for dim in [128, 256, 512, 1024]:
        errs1 = []
        errs2 = []
        errs3 = []
        relerrs1 = []
        relerrs2 = []
        relerrs3 = []
        max_errs1 = []
        max_errs2 = []
        max_errs3 = []
        for i in range(100):
            if kind == 'fc1':
                A = torch.randn(1, dim, dtype=dtype, device='cuda')
                B = torch.randn(dim * 4, dim, dtype=dtype, device='cuda') / math.sqrt(dim)
            elif kind == 'fc2':
                A = torch.randn(1, 4 * dim, dtype=dtype, device='cuda')
                B = torch.randn(dim, 4 * dim, dtype=dtype, device='cuda') / math.sqrt(dim)
            elif kind == 'attn':
                A = torch.randn(1, dim, dtype=dtype, device='cuda')
                B = torch.randn(dim, dim, dtype=dtype, device='cuda') / math.sqrt(dim)
            elif kind == 'attn_packed':
                A = torch.randn(1, dim, dtype=dtype, device='cuda')
                B = torch.randn(dim * 3, dim, dtype=dtype, device='cuda') / math.sqrt(dim)
            qB, state = F.quantize_4bit(B, quant_type=storage_type, compress_statistics=double_quant, quant_storage=quant_storage)
            C3 = torch.matmul(A, B.t())
            C2 = F.gemv_4bit(A, qB.t(), state=state)
            A.requires_grad = True
            C1 = bnb.matmul_4bit(A, qB.t(), state)
            err1 = (C1 - C2).abs().float()
            err2 = (C3 - C2).abs().float()
            err3 = (C3 - C1).abs().float()
            mag1 = torch.abs(C1).float() + 1e-05
            mag2 = torch.abs(C3).float() + 1e-05
            mag3 = torch.abs(C3).float() + 1e-05
            relerr1 = err1 / mag1
            relerr2 = err2 / mag2
            relerr3 = err3 / mag3
            max_err1 = err1.max()
            max_err2 = err2.max()
            max_err3 = err3.max()
            errs1.append(err1.mean().item())
            errs2.append(err2.mean().item())
            errs3.append(err3.mean().item())
            relerrs1.append(relerr1.mean().item())
            relerrs2.append(relerr2.mean().item())
            relerrs3.append(relerr3.mean().item())
            max_errs1.append(max_err1.item())
            max_errs2.append(max_err2.item())
            max_errs3.append(max_err3.item())
            c = int(C1.numel() * 0.0014 * (dim / 256)) + 1
            c = assert_all_approx_close(C1, C2, 1e-05, 0.01, count=c, throw=False)
        err1 = sum(errs1) / len(errs1) / math.sqrt(dim)
        err2 = sum(errs2) / len(errs2) / math.sqrt(dim)
        err3 = sum(errs3) / len(errs3) / math.sqrt(dim)
        relerr1 = sum(relerrs1) / len(relerrs1) / math.sqrt(dim)
        relerr2 = sum(relerrs2) / len(relerrs2) / math.sqrt(dim)
        relerr3 = sum(relerrs3) / len(relerrs3) / math.sqrt(dim)
        maxerr1 = sum(max_errs1) / len(max_errs1) / math.sqrt(dim)
        maxerr2 = sum(max_errs2) / len(max_errs2) / math.sqrt(dim)
        maxerr3 = sum(max_errs3) / len(max_errs3) / math.sqrt(dim)
        absratio = err2 / err3
        relratio = relerr2 / relerr3
        maxratio = relerr2 / relerr3
        if dtype == torch.float16:
            if dim <= 512:
                assert err1 < 7e-05
                assert relerr1 < 0.0008
            else:
                assert err1 < 6e-05
                assert relerr1 < 0.0002
            assert absratio < 1.005 and absratio > 0.995
            assert relratio < 1.005 and relratio > 0.995
            assert maxratio < 1.005 and maxratio > 0.995
        elif dtype == torch.float32:
            if dim <= 512:
                assert err1 < 5e-08
                assert relerr1 < 1e-06
                assert maxerr1 < 1e-07
            else:
                assert err1 < 5e-08
                assert relerr1 < 8e-06
                assert maxerr1 < 1e-07
            assert absratio < 1.005 and absratio > 0.995
            assert relratio < 1.005 and relratio > 0.995
            assert maxratio < 1.005 and maxratio > 0.995
        elif dtype == torch.bfloat16:
            if dim <= 512:
                assert err1 < 0.0006
                assert relerr1 < 0.007
                assert maxerr1 < 0.015
            else:
                assert err1 < 0.0002
                assert relerr1 < 0.002
                assert maxerr1 < 0.0012
            assert absratio < 1.005 and absratio > 0.995
            assert relratio < 1.04 and relratio > 0.96
            assert maxratio < 1.02 and maxratio > 0.98