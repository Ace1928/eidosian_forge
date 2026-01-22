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
@pytest.mark.parametrize('dim1', [1 * 1024], ids=id_formatter('dim1'))
@pytest.mark.parametrize('dim2', [1 * 1024], ids=id_formatter('dim2'))
@pytest.mark.parametrize('dims', (2,), ids=id_formatter('dims'))
def test_colrow_absmax(dim1, dim2, dims):
    for i in range(k):
        threshold = 3.0
        A = torch.randn(dim1, dim2, device='cuda').half()
        A_truncated = A.clone()
        A_truncated[torch.abs(A_truncated) >= 3.0] = 0.0
        if dims == 2:
            row_stats1, _ = torch.abs(A.float()).max(1)
            col_stats1, _ = torch.abs(A.float()).max(0)
            row_stats1_trunc, _ = torch.abs(A_truncated.float()).max(1)
            col_stats1_trunc, _ = torch.abs(A_truncated.float()).max(0)
        else:
            assert False
        row_stats2, col_stats2, nnz_block_ptr2 = F.get_colrow_absmax(A, threshold=threshold)
        A_blocked = einops.rearrange(torch.abs(A), '(rows row_tiles) (cols block_size)-> rows cols row_tiles block_size', row_tiles=16, block_size=64 * 4)
        nnz_rows1_counts = (torch.abs(A_blocked) >= threshold).sum(3).flatten()
        nnz_block_ptr1 = torch.zeros(nnz_rows1_counts.shape[0] + 1, dtype=nnz_rows1_counts.dtype, device=nnz_rows1_counts.device)
        nnz_block_ptr1[1:] = nnz_rows1_counts.cumsum(0)
        torch.testing.assert_close(col_stats1_trunc, col_stats2)
        torch.testing.assert_close(row_stats1_trunc, row_stats2)
        torch.testing.assert_close(nnz_block_ptr1.int(), nnz_block_ptr2)
        row_stats2, col_stats2, nnz_block_ptr2 = F.get_colrow_absmax(A, threshold=0.0)
        torch.testing.assert_close(col_stats1, col_stats2)
        torch.testing.assert_close(row_stats1, row_stats2)
        assert nnz_block_ptr2 is None