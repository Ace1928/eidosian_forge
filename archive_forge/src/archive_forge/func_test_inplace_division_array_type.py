import sys
import warnings
import copy
import operator
import itertools
import textwrap
import pytest
from functools import reduce
import numpy as np
import numpy.ma.core
import numpy.core.fromnumeric as fromnumeric
import numpy.core.umath as umath
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy import ndarray
from numpy.compat import asbytes
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.compat import pickle
def test_inplace_division_array_type(self):
    for t in self.othertypes:
        with suppress_warnings() as sup:
            sup.record(UserWarning)
            x, y, xm = (_.astype(t) for _ in self.uint8data)
            m = xm.mask
            a = arange(10, dtype=t)
            a[-1] = masked
            try:
                x /= a
                assert_equal(x, y / a)
            except (DeprecationWarning, TypeError) as e:
                warnings.warn(str(e), stacklevel=1)
            try:
                xm /= a
                assert_equal(xm, y / a)
                assert_equal(xm.mask, mask_or(mask_or(m, a.mask), a == t(0)))
            except (DeprecationWarning, TypeError) as e:
                warnings.warn(str(e), stacklevel=1)
            if issubclass(t, np.integer):
                assert_equal(len(sup.log), 2, f'Failed on type={t}.')
            else:
                assert_equal(len(sup.log), 0, f'Failed on type={t}.')