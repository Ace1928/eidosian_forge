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
@pytest.mark.parametrize('hidden_dim', get_test_dims(32, 256, n=2), ids=id_formatter('hidden_dim'))
@pytest.mark.parametrize('batch_dim', get_test_dims(16, 256, n=2), ids=id_formatter('batch_dim'))
@pytest.mark.parametrize('seq_dim', get_test_dims(16, 256, n=2), ids=id_formatter('seq_dim'))
@pytest.mark.parametrize('transpose', BOOLEAN_TUPLES, ids=id_formatter('transpose'))
def test_igemm(hidden_dim, batch_dim, transpose, seq_dim):
    hidden_dim = hidden_dim - hidden_dim % 32
    batch_dim = batch_dim - batch_dim % 16
    seq_dim = seq_dim - seq_dim % 16
    for i in range(k):
        shapeA = (batch_dim, hidden_dim) if not transpose[0] else (hidden_dim, batch_dim)
        shapeB = (32 * random.randint(1, 4), hidden_dim) if transpose[1] else (hidden_dim, 32 * random.randint(1, 4))
        A = torch.randint(-128, 127, size=shapeA, device='cuda').to(torch.int8)
        B = torch.randint(-128, 127, size=shapeB, device='cuda').to(torch.int8)
        if not transpose[0] and (not transpose[1]):
            out2 = torch.matmul(A.float(), B.float())
            out = F.igemm(A, B)
        elif not transpose[0] and transpose[1]:
            out2 = torch.matmul(A.float(), B.t().float())
            out = F.igemm(A, B.t())
        elif transpose[0] and (not transpose[1]):
            out2 = torch.matmul(A.t().float(), B.float())
            out = F.igemm(A.t(), B)
        elif transpose[0] and transpose[1]:
            out2 = torch.matmul(A.t().float(), B.t().float())
            out = F.igemm(A.t(), B.t())
        torch.testing.assert_close(out.float(), out2)
    for i in range(k):
        shapeA = (batch_dim, seq_dim, hidden_dim)
        shapeB = (32 * random.randint(1, 4), hidden_dim) if transpose[1] else (hidden_dim, 32 * random.randint(1, 4))
        A = torch.randint(-128, 127, size=shapeA, device='cuda').to(torch.int8)
        B = torch.randint(-128, 127, size=shapeB, device='cuda').to(torch.int8)
        if not transpose[0] and (not transpose[1]):
            out2 = torch.matmul(A.float(), B.float())
            out = F.igemm(A, B)
        elif not transpose[0] and transpose[1]:
            out2 = torch.matmul(A.float(), B.t().float())
            out = F.igemm(A, B.t())
        torch.testing.assert_close(out.float(), out2)