from __future__ import annotations
from contextlib import suppress
import numpy as np
import pytest
from xarray import Variable
from xarray.coding import strings
from xarray.core import indexing
from xarray.tests import (
@requires_dask
def test_bytes_to_char_dask() -> None:
    numpy_array = np.array([b'ab', b'cd'])
    array = da.from_array(numpy_array, ((1, 1),))
    expected = np.array([[b'a', b'b'], [b'c', b'd']])
    actual = strings.bytes_to_char(array)
    assert isinstance(actual, da.Array)
    assert actual.chunks == ((1, 1), (2,))
    assert actual.dtype == 'S1'
    assert_array_equal(np.array(actual), expected)