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
def test_where_type(self):
    x = np.arange(4, dtype=np.int32)
    y = np.arange(4, dtype=np.float32) * 2.2
    test = where(x > 1.5, y, x).dtype
    control = np.result_type(np.int32, np.float32)
    assert_equal(test, control)