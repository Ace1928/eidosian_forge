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
def test_inplace_floor_division_scalar_type(self):
    unsupported = {np.dtype(t).type for t in np.typecodes['Complex']}
    for t in self.othertypes:
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            x, y, xm = (_.astype(t) for _ in self.uint8data)
            x = arange(10, dtype=t) * t(2)
            xm = arange(10, dtype=t) * t(2)
            xm[2] = masked
            try:
                x //= t(2)
                xm //= t(2)
                assert_equal(x, y)
                assert_equal(xm, y)
            except TypeError:
                msg = f'Supported type {t} throwing TypeError'
                assert t in unsupported, msg