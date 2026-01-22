import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
def test_masked_all(self):
    test = masked_all((2,), dtype=float)
    control = array([1, 1], mask=[1, 1], dtype=float)
    assert_equal(test, control)
    dt = np.dtype({'names': ['a', 'b'], 'formats': ['f', 'f']})
    test = masked_all((2,), dtype=dt)
    control = array([(0, 0), (0, 0)], mask=[(1, 1), (1, 1)], dtype=dt)
    assert_equal(test, control)
    test = masked_all((2, 2), dtype=dt)
    control = array([[(0, 0), (0, 0)], [(0, 0), (0, 0)]], mask=[[(1, 1), (1, 1)], [(1, 1), (1, 1)]], dtype=dt)
    assert_equal(test, control)
    dt = np.dtype([('a', 'f'), ('b', [('ba', 'f'), ('bb', 'f')])])
    test = masked_all((2,), dtype=dt)
    control = array([(1, (1, 1)), (1, (1, 1))], mask=[(1, (1, 1)), (1, (1, 1))], dtype=dt)
    assert_equal(test, control)
    test = masked_all((2,), dtype=dt)
    control = array([(1, (1, 1)), (1, (1, 1))], mask=[(1, (1, 1)), (1, (1, 1))], dtype=dt)
    assert_equal(test, control)
    test = masked_all((1, 1), dtype=dt)
    control = array([[(1, (1, 1))]], mask=[[(1, (1, 1))]], dtype=dt)
    assert_equal(test, control)