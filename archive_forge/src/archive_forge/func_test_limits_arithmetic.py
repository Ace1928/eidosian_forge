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
def test_limits_arithmetic(self):
    tiny = np.finfo(float).tiny
    a = array([tiny, 1.0 / tiny, 0.0])
    assert_equal(getmaskarray(a / 2), [0, 0, 0])
    assert_equal(getmaskarray(2 / a), [1, 0, 1])