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
@pytest.mark.parametrize(('shape', 'chunk'), [(20, 10), (50, 10), (70, 20)])
def test_solve_triangular_vector(shape, chunk):
    rng = np.random.default_rng(1)
    A = rng.integers(1, 11, (shape, shape))
    b = rng.integers(1, 11, shape)
    Au = np.triu(A)
    dAu = da.from_array(Au, (chunk, chunk))
    db = da.from_array(b, chunk)
    res = da.linalg.solve_triangular(dAu, db)
    assert_eq(res, scipy.linalg.solve_triangular(Au, b))
    assert_eq(dAu.dot(res), b.astype(float))
    Al = np.tril(A)
    dAl = da.from_array(Al, (chunk, chunk))
    db = da.from_array(b, chunk)
    res = da.linalg.solve_triangular(dAl, db, lower=True)
    assert_eq(res, scipy.linalg.solve_triangular(Al, b, lower=True))
    assert_eq(dAl.dot(res), b.astype(float))