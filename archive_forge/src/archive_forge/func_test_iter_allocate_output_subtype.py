import pytest
import textwrap
import warnings
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_raises,
def test_iter_allocate_output_subtype():
    a = np.matrix([[1, 2], [3, 4]])
    b = np.arange(4).reshape(2, 2).T
    i = np.nditer([a, b, None], [], [['readonly'], ['readonly'], ['writeonly', 'allocate']])
    assert_(type(i.operands[2]) is np.matrix)
    assert_(type(i.operands[2]) is not np.ndarray)
    assert_equal(i.operands[2].shape, (2, 2))
    b = np.arange(4).reshape(1, 2, 2)
    assert_raises(RuntimeError, np.nditer, [a, b, None], [], [['readonly'], ['readonly'], ['writeonly', 'allocate']])
    i = np.nditer([a, b, None], [], [['readonly'], ['readonly'], ['writeonly', 'allocate', 'no_subtype']])
    assert_(type(i.operands[2]) is np.ndarray)
    assert_(type(i.operands[2]) is not np.matrix)
    assert_equal(i.operands[2].shape, (1, 2, 2))