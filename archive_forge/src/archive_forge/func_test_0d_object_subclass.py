import sys
import gc
from hypothesis import given
from hypothesis.extra import numpy as hynp
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.arrayprint import _typelessdata
import textwrap
def test_0d_object_subclass(self):

    class sub(np.ndarray):

        def __new__(cls, inp):
            obj = np.asarray(inp).view(cls)
            return obj

        def __getitem__(self, ind):
            ret = super().__getitem__(ind)
            return sub(ret)
    x = sub(1)
    assert_equal(repr(x), 'sub(1)')
    assert_equal(str(x), '1')
    x = sub([1, 1])
    assert_equal(repr(x), 'sub([1, 1])')
    assert_equal(str(x), '[1 1]')
    x = sub(None)
    assert_equal(repr(x), 'sub(None, dtype=object)')
    assert_equal(str(x), 'None')
    y = sub(None)
    x[()] = y
    y[()] = x
    assert_equal(repr(x), 'sub(sub(sub(..., dtype=object), dtype=object), dtype=object)')
    assert_equal(str(x), '...')
    x[()] = 0
    x = sub(None)
    x[()] = sub(None)
    assert_equal(repr(x), 'sub(sub(None, dtype=object), dtype=object)')
    assert_equal(str(x), 'None')

    class DuckCounter(np.ndarray):

        def __getitem__(self, item):
            result = super().__getitem__(item)
            if not isinstance(result, DuckCounter):
                result = result[...].view(DuckCounter)
            return result

        def to_string(self):
            return {0: 'zero', 1: 'one', 2: 'two'}.get(self.item(), 'many')

        def __str__(self):
            if self.shape == ():
                return self.to_string()
            else:
                fmt = {'all': lambda x: x.to_string()}
                return np.array2string(self, formatter=fmt)
    dc = np.arange(5).view(DuckCounter)
    assert_equal(str(dc), '[zero one two many many]')
    assert_equal(str(dc[0]), 'zero')