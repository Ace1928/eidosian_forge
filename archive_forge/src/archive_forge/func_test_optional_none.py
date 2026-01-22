import os
import sys
import copy
import platform
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal
from numpy.core.multiarray import typeinfo as _typeinfo
from . import util
def test_optional_none(self):
    shape = (2,)
    a = self.array(shape, intent.optional, None)
    assert a.arr.shape == shape
    assert a.arr_equal(a.arr, np.zeros(shape, dtype=self.type.dtype))
    shape = (2, 3)
    a = self.array(shape, intent.optional, None)
    assert a.arr.shape == shape
    assert a.arr_equal(a.arr, np.zeros(shape, dtype=self.type.dtype))
    assert a.arr.flags['FORTRAN'] and (not a.arr.flags['CONTIGUOUS'])
    shape = (2, 3)
    a = self.array(shape, intent.c.optional, None)
    assert a.arr.shape == shape
    assert a.arr_equal(a.arr, np.zeros(shape, dtype=self.type.dtype))
    assert not a.arr.flags['FORTRAN'] and a.arr.flags['CONTIGUOUS']