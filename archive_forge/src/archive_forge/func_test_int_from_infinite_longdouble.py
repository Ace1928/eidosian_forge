import contextlib
import sys
import warnings
import itertools
import operator
import platform
from numpy._utils import _pep440
import pytest
from hypothesis import given, settings
from hypothesis.strategies import sampled_from
from hypothesis.extra import numpy as hynp
import numpy as np
from numpy.testing import (
def test_int_from_infinite_longdouble(self):
    x = np.longdouble(np.inf)
    assert_raises(OverflowError, int, x)
    with suppress_warnings() as sup:
        sup.record(np.ComplexWarning)
        x = np.clongdouble(np.inf)
        assert_raises(OverflowError, int, x)
        assert_equal(len(sup.log), 1)