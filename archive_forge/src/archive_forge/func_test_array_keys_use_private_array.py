import operator
from numpy.testing import assert_raises, suppress_warnings
import numpy as np
import pytest
from .. import ones, asarray, reshape, result_type, all, equal
from .._array_object import Array
from .._dtypes import (
def test_array_keys_use_private_array():
    """
    Indexing operations convert array keys before indexing the internal array

    Fails when array_api array keys are not converted into NumPy-proper arrays
    in __getitem__(). This is achieved by passing array_api arrays with 0-sized
    dimensions, which NumPy-proper treats erroneously - not sure why!

    TODO: Find and use appropriate __setitem__() case.
    """
    a = ones((0, 0), dtype=bool_)
    assert a[a].shape == (0,)
    a = ones((0,), dtype=bool_)
    key = ones((0, 0), dtype=bool_)
    with pytest.raises(IndexError):
        a[key]