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
@pytest.mark.parametrize('shape, chunks, axis', [[(5,), (2,), None], [(5,), (2,), 0], [(5,), (2,), (0,)]])
@pytest.mark.parametrize('norm', [0, 2, -2, 0.5])
@pytest.mark.parametrize('keepdims', [False, True])
def test_norm_1dim(shape, chunks, axis, norm, keepdims):
    a = np.random.default_rng().random(shape)
    d = da.from_array(a, chunks=chunks)
    a_r = np.linalg.norm(a, ord=norm, axis=axis, keepdims=keepdims)
    d_r = da.linalg.norm(d, ord=norm, axis=axis, keepdims=keepdims)
    assert_eq(a_r, d_r)