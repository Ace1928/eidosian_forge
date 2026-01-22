from __future__ import annotations
import sys
import pytest
import numpy as np
import scipy.linalg
from packaging.version import parse as parse_version
import dask.array as da
from dask.array.linalg import qr, sfqr, svd, svd_compressed, tsqr
from dask.array.numpy_compat import _np_version
from dask.array.utils import assert_eq, same_keys, svd_flip
@pytest.mark.slow
@pytest.mark.parametrize('size', [10, 20, 30, 50])
@pytest.mark.filterwarnings('ignore:Increasing:dask.array.core.PerformanceWarning')
def test_lu_2(size):
    rng = np.random.default_rng(10)
    A = rng.integers(0, 10, (size, size))
    dA = da.from_array(A, chunks=(5, 5))
    dp, dl, du = da.linalg.lu(dA)
    _check_lu_result(dp, dl, du, A)