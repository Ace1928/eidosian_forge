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
def test_arraymethod(self):
    marray = masked_array([[1, 2, 3, 4, 5]], mask=[0, 0, 1, 0, 0])
    control = masked_array([[1], [2], [3], [4], [5]], mask=[0, 0, 1, 0, 0])
    assert_equal(marray.T, control)
    assert_equal(marray.transpose(), control)
    assert_equal(MaskedArray.cumsum(marray.T, 0), control.cumsum(0))