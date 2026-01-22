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
def test_dask_token():
    import dask
    s = sparse.COO.from_numpy(np.array([0, 0, 1, 2]))
    a = DataArray(s)
    t1 = dask.base.tokenize(a)
    t2 = dask.base.tokenize(a)
    t3 = dask.base.tokenize(a + 1)
    assert t1 == t2
    assert t3 != t2
    assert isinstance(a.data, sparse.COO)
    ac = a.chunk(2)
    t4 = dask.base.tokenize(ac)
    t5 = dask.base.tokenize(ac + 1)
    assert t4 != t5
    assert isinstance(ac.data._meta, sparse.COO)