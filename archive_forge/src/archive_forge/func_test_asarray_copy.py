from numpy.testing import assert_raises
import numpy as np
from .. import all
from .._creation_functions import (
from .._dtypes import float32, float64
from .._array_object import Array
def test_asarray_copy():
    a = asarray([1])
    b = asarray(a, copy=True)
    a[0] = 0
    assert all(b[0] == 1)
    assert all(a[0] == 0)
    a = asarray([1])
    b = asarray(a, copy=np._CopyMode.ALWAYS)
    a[0] = 0
    assert all(b[0] == 1)
    assert all(a[0] == 0)
    a = asarray([1])
    b = asarray(a, copy=np._CopyMode.NEVER)
    a[0] = 0
    assert all(b[0] == 0)
    assert_raises(NotImplementedError, lambda: asarray(a, copy=False))
    assert_raises(NotImplementedError, lambda: asarray(a, copy=np._CopyMode.IF_NEEDED))