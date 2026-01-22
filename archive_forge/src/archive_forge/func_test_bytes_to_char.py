from __future__ import annotations
from contextlib import suppress
import numpy as np
import pytest
from xarray import Variable
from xarray.coding import strings
from xarray.core import indexing
from xarray.tests import (
def test_bytes_to_char() -> None:
    array = np.array([[b'ab', b'cd'], [b'ef', b'gh']])
    expected = np.array([[[b'a', b'b'], [b'c', b'd']], [[b'e', b'f'], [b'g', b'h']]])
    actual = strings.bytes_to_char(array)
    assert_array_equal(actual, expected)
    expected = np.array([[[b'a', b'b'], [b'e', b'f']], [[b'c', b'd'], [b'g', b'h']]])
    actual = strings.bytes_to_char(array.T)
    assert_array_equal(actual, expected)