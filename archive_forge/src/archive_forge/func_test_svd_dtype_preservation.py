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
@pytest.mark.parametrize('chunks', [(10, 50), (50, 10), (-1, -1)])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_svd_dtype_preservation(chunks, dtype):
    x = da.random.default_rng().random((50, 50), chunks=chunks).astype(dtype)
    u, s, v = svd(x)
    assert u.dtype == s.dtype == v.dtype == dtype