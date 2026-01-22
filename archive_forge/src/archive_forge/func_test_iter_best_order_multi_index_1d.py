import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_best_order_multi_index_1d():
    a = arange(4)
    i = nditer(a, ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i), [(0,), (1,), (2,), (3,)])
    i = nditer(a[::-1], ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i), [(3,), (2,), (1,), (0,)])