import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_non_writable_attribute_deletion():
    it = np.nditer(np.ones(2))
    attr = ['value', 'shape', 'operands', 'itviews', 'has_delayed_bufalloc', 'iterationneedsapi', 'has_multi_index', 'has_index', 'dtypes', 'ndim', 'nop', 'itersize', 'finished']
    for s in attr:
        assert_raises(AttributeError, delattr, it, s)