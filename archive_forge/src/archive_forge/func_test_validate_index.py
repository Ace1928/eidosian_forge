import operator
from numpy.testing import assert_raises, suppress_warnings
import numpy as np
import pytest
from .. import ones, asarray, reshape, result_type, all, equal
from .._array_object import Array
from .._dtypes import (
def test_validate_index():
    a = ones((3, 4))
    assert_raises(IndexError, lambda: a[:4])
    assert_raises(IndexError, lambda: a[:-4])
    assert_raises(IndexError, lambda: a[:3:-1])
    assert_raises(IndexError, lambda: a[:-5:-1])
    assert_raises(IndexError, lambda: a[4:])
    assert_raises(IndexError, lambda: a[-4:])
    assert_raises(IndexError, lambda: a[4::-1])
    assert_raises(IndexError, lambda: a[-4::-1])
    assert_raises(IndexError, lambda: a[..., :5])
    assert_raises(IndexError, lambda: a[..., :-5])
    assert_raises(IndexError, lambda: a[..., :5:-1])
    assert_raises(IndexError, lambda: a[..., :-6:-1])
    assert_raises(IndexError, lambda: a[..., 5:])
    assert_raises(IndexError, lambda: a[..., -5:])
    assert_raises(IndexError, lambda: a[..., 5::-1])
    assert_raises(IndexError, lambda: a[..., -5::-1])
    assert_raises(IndexError, lambda: a[a[:, 0] == 1, 0])
    assert_raises(IndexError, lambda: a[a[:, 0] == 1, ...])
    assert_raises(IndexError, lambda: a[..., a[0] == 1])
    assert_raises(IndexError, lambda: a[[True, True, True]])
    assert_raises(IndexError, lambda: a[(True, True, True),])
    idx = asarray([[0, 1]])
    assert_raises(IndexError, lambda: a[idx])
    assert_raises(IndexError, lambda: a[idx,])
    assert_raises(IndexError, lambda: a[[0, 1]])
    assert_raises(IndexError, lambda: a[(0, 1), (0, 1)])
    assert_raises(IndexError, lambda: a[[0, 1]])
    assert_raises(IndexError, lambda: a[np.array([[0, 1]])])
    assert_raises(IndexError, lambda: a[()])
    assert_raises(IndexError, lambda: a[0,])
    assert_raises(IndexError, lambda: a[0])
    assert_raises(IndexError, lambda: a[:])