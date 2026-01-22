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
@pytest.mark.parametrize('chunks', [(10, -1), (-1, 10), (9, -1), (-1, 9)])
@pytest.mark.parametrize('shape', [(10, 100), (100, 10), (10, 10)])
def test_svd_supported_array_shapes(chunks, shape):
    x = np.random.default_rng().random(shape)
    dx = da.from_array(x, chunks=chunks)
    du, ds, dv = da.linalg.svd(dx)
    du, dv = da.compute(du, dv)
    nu, ns, nv = np.linalg.svd(x, full_matrices=False)
    du, dv = svd_flip(du, dv)
    nu, nv = svd_flip(nu, nv)
    assert_eq(du, nu)
    assert_eq(ds, ns)
    assert_eq(dv, nv)