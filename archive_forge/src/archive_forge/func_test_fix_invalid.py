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
def test_fix_invalid(self):
    with np.errstate(invalid='ignore'):
        data = masked_array([np.nan, 0.0, 1.0], mask=[0, 0, 1])
        data_fixed = fix_invalid(data)
        assert_equal(data_fixed._data, [data.fill_value, 0.0, 1.0])
        assert_equal(data_fixed._mask, [1.0, 0.0, 1.0])