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
def test_large_types(self):
    for t in [np.int32, np.int64, np.float32, np.float64, np.longdouble]:
        a = t(51)
        b = a ** 4
        msg = 'error with %r: got %r' % (t, b)
        if np.issubdtype(t, np.integer):
            assert_(b == 6765201, msg)
        else:
            assert_almost_equal(b, 6765201, err_msg=msg)