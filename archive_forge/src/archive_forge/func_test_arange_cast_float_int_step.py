from __future__ import annotations
import pytest
import numpy as np
import pytest
from tlz import concat
import dask
import dask.array as da
from dask.array.core import normalize_chunks
from dask.array.numpy_compat import AxisError
from dask.array.utils import assert_eq, same_keys
@pytest.mark.xfail(reason='Casting floats to ints is not supported since edgebehavior is not specified or guaranteed by NumPy.')
def test_arange_cast_float_int_step():
    darr = da.arange(3.3, -9.1, -0.25, chunks=3, dtype='i8')
    nparr = np.arange(3.3, -9.1, -0.25, dtype='i8')
    assert_eq(darr, nparr)