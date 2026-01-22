from __future__ import annotations
from contextlib import suppress
import numpy as np
import pytest
from xarray import Variable
from xarray.coding import strings
from xarray.core import indexing
from xarray.tests import (
def test_char_to_bytes_ndim_zero() -> None:
    expected = np.array(b'a')
    actual = strings.char_to_bytes(expected)
    assert_array_equal(actual, expected)