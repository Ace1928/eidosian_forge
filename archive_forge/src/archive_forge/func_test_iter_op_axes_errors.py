import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_op_axes_errors():
    a = arange(6).reshape(2, 3)
    assert_raises(ValueError, nditer, [a, a], [], [['readonly']] * 2, op_axes=[[0], [1], [0]])
    assert_raises(ValueError, nditer, [a, a], [], [['readonly']] * 2, op_axes=[[2, 1], [0, 1]])
    assert_raises(ValueError, nditer, [a, a], [], [['readonly']] * 2, op_axes=[[0, 1], [2, -1]])
    assert_raises(ValueError, nditer, [a, a], [], [['readonly']] * 2, op_axes=[[0, 0], [0, 1]])
    assert_raises(ValueError, nditer, [a, a], [], [['readonly']] * 2, op_axes=[[0, 1], [1, 1]])
    assert_raises(ValueError, nditer, [a, a], [], [['readonly']] * 2, op_axes=[[0, 1], [0, 1, 0]])
    assert_raises(ValueError, nditer, [a, a], [], [['readonly']] * 2, op_axes=[[0, 1], [1, 0]])