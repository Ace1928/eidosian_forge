import sys
import gc
from hypothesis import given
from hypothesis.extra import numpy as hynp
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.arrayprint import _typelessdata
import textwrap
def test_formatter_reset(self):
    x = np.arange(3)
    np.set_printoptions(formatter={'all': lambda x: str(x - 1)})
    assert_equal(repr(x), 'array([-1, 0, 1])')
    np.set_printoptions(formatter={'int': None})
    assert_equal(repr(x), 'array([0, 1, 2])')
    np.set_printoptions(formatter={'all': lambda x: str(x - 1)})
    assert_equal(repr(x), 'array([-1, 0, 1])')
    np.set_printoptions(formatter={'all': None})
    assert_equal(repr(x), 'array([0, 1, 2])')
    np.set_printoptions(formatter={'int': lambda x: str(x - 1)})
    assert_equal(repr(x), 'array([-1, 0, 1])')
    np.set_printoptions(formatter={'int_kind': None})
    assert_equal(repr(x), 'array([0, 1, 2])')
    x = np.arange(3.0)
    np.set_printoptions(formatter={'float': lambda x: str(x - 1)})
    assert_equal(repr(x), 'array([-1.0, 0.0, 1.0])')
    np.set_printoptions(formatter={'float_kind': None})
    assert_equal(repr(x), 'array([0., 1., 2.])')