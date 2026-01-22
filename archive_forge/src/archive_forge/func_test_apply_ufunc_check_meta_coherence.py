from __future__ import annotations
import math
import pickle
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Variable
from xarray.namedarray.pycompat import array_type
from xarray.tests import assert_equal, assert_identical, requires_dask
@requires_dask
def test_apply_ufunc_check_meta_coherence():
    s = sparse.COO.from_numpy(np.array([0, 0, 1, 2]))
    a = DataArray(s)
    ac = a.chunk(2)
    sparse_meta = ac.data._meta
    result = xr.apply_ufunc(lambda x: x, ac, dask='parallelized').data._meta
    assert_sparse_equal(result, sparse_meta)