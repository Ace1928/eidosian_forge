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
def test_masked_invalid_pandas(self):

    class Series:
        _data = 'nonsense'

        def __array__(self):
            return np.array([5, np.nan, np.inf])
    arr = np.ma.masked_invalid(Series())
    assert_array_equal(arr._data, np.array(Series()))
    assert_array_equal(arr._mask, [False, True, True])