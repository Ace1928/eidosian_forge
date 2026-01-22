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
def test_svd_incompatible_chunking():
    with pytest.raises(NotImplementedError, match='Array must be chunked in one dimension only'):
        x = da.random.default_rng().random((10, 10), chunks=(5, 5))
        da.linalg.svd(x)