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
@pytest.mark.parametrize('ref', [1, [1], 'MyNoArrayFunctionArray'])
def test_no_array_function_like(self, function, args, kwargs, ref):
    self.add_method('array', self.MyNoArrayFunctionArray)
    self.add_method(function, self.MyNoArrayFunctionArray)
    np_func = getattr(np, function)
    if ref == 'MyNoArrayFunctionArray':
        ref = self.MyNoArrayFunctionArray.array()
    like_args = tuple((a() if callable(a) else a for a in args))
    with assert_raises_regex(TypeError, 'The `like` argument must be an array-like that implements'):
        np_func(*like_args, **kwargs, like=ref)