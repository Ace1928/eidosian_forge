import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
def test_argequivalent(self):
    """ Test it translates from arg<func> to <func> """
    from numpy.random import rand
    a = rand(3, 4, 5)
    funcs = [(np.sort, np.argsort, dict()), (_add_keepdims(np.min), _add_keepdims(np.argmin), dict()), (_add_keepdims(np.max), _add_keepdims(np.argmax), dict()), (np.partition, np.argpartition, dict(kth=2))]
    for func, argfunc, kwargs in funcs:
        for axis in list(range(a.ndim)) + [None]:
            a_func = func(a, axis=axis, **kwargs)
            ai_func = argfunc(a, axis=axis, **kwargs)
            assert_equal(a_func, take_along_axis(a, ai_func, axis=axis))