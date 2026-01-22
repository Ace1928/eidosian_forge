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
def test_no_chunks_svd():
    x = np.random.default_rng().random((100, 10))
    u, s, v = np.linalg.svd(x, full_matrices=False)
    for chunks in [((np.nan,) * 10, (10,)), ((np.nan,) * 10, (np.nan,))]:
        dx = da.from_array(x, chunks=(10, 10))
        dx._chunks = chunks
        du, ds, dv = da.linalg.svd(dx)
        assert_eq(s, ds)
        assert_eq(u.dot(np.diag(s)).dot(v), du.dot(da.diag(ds)).dot(dv))
        assert_eq(du.T.dot(du), np.eye(10))
        assert_eq(dv.T.dot(dv), np.eye(10))
        dx = da.from_array(x, chunks=(10, 10))
        dx._chunks = ((np.nan,) * 10, (np.nan,))
        assert_eq(abs(v), abs(dv))
        assert_eq(abs(u), abs(du))