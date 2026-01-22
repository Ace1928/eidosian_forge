from __future__ import annotations
import pytest
import numpy as np
import dask
import dask.array as da
from dask.array.core import Array
from dask.array.utils import assert_eq
from dask.multiprocessing import _dumps, _loads
from dask.utils import key_split
@pytest.mark.parametrize('sz', [None, 5, (2, 2)], ids=type)
def test_Generator_only_funcs(sz):
    da.random.default_rng().integers(5, high=15, size=sz, chunks=3).compute()
    da.random.default_rng().multivariate_hypergeometric([16, 8, 4], 6, size=sz, chunks=6).compute()