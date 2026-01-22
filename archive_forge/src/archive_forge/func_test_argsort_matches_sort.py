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
def test_argsort_matches_sort(self):
    x = array([1, 4, 2, 3], mask=[0, 1, 0, 0], dtype=np.uint8)
    for kwargs in [dict(), dict(endwith=True), dict(endwith=False), dict(fill_value=2), dict(fill_value=2, endwith=True), dict(fill_value=2, endwith=False)]:
        sortedx = sort(x, **kwargs)
        argsortedx = x[argsort(x, **kwargs)]
        assert_equal(sortedx._data, argsortedx._data)
        assert_equal(sortedx._mask, argsortedx._mask)