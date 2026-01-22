import inspect
import sys
import os
import tempfile
from io import StringIO
from unittest import mock
import numpy as np
from numpy.testing import (
from numpy.core.overrides import (
from numpy.compat import pickle
import pytest
@pytest.mark.parametrize('function, args, kwargs', _array_tests)
def test_like_as_none(self, function, args, kwargs):
    self.add_method('array', self.MyArray)
    self.add_method(function, self.MyArray)
    np_func = getattr(np, function)
    like_args = tuple((a() if callable(a) else a for a in args))
    like_args_exp = tuple((a() if callable(a) else a for a in args))
    array_like = np_func(*like_args, **kwargs, like=None)
    expected = np_func(*like_args_exp, **kwargs)
    if function == 'empty':
        array_like.fill(1)
        expected.fill(1)
    assert_equal(array_like, expected)