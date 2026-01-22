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
@pytest.mark.parametrize('shape, chunks, pad_width, mode, kwargs', [((10, 11), (4, 5), 0, 'constant', {'constant_values': 2}), ((10, 11), (4, 5), 0, 'edge', {}), ((10, 11), (4, 5), 0, 'linear_ramp', {'end_values': 2}), ((10, 11), (4, 5), 0, 'reflect', {}), ((10, 11), (4, 5), 0, 'symmetric', {}), ((10, 11), (4, 5), 0, 'wrap', {}), ((10, 11), (4, 5), 0, 'empty', {})])
def test_pad_0_width(shape, chunks, pad_width, mode, kwargs):
    np_a = np.random.random(shape)
    da_a = da.from_array(np_a, chunks=chunks)
    np_r = np.pad(np_a, pad_width, mode, **kwargs)
    da_r = da.pad(da_a, pad_width, mode, **kwargs)
    assert da_r is da_a
    assert_eq(np_r, da_r)