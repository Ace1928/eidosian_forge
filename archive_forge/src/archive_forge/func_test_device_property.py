import operator
from numpy.testing import assert_raises, suppress_warnings
import numpy as np
import pytest
from .. import ones, asarray, reshape, result_type, all, equal
from .._array_object import Array
from .._dtypes import (
def test_device_property():
    a = ones((3, 4))
    assert a.device == 'cpu'
    assert all(equal(a.to_device('cpu'), a))
    assert_raises(ValueError, lambda: a.to_device('gpu'))
    assert all(equal(asarray(a, device='cpu'), a))
    assert_raises(ValueError, lambda: asarray(a, device='gpu'))