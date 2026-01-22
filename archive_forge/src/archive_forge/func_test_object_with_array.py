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
def test_object_with_array(self):
    mx1 = masked_array([1.0], mask=[True])
    mx2 = masked_array([1.0, 2.0])
    mx = masked_array([mx1, mx2], mask=[False, True], dtype=object)
    assert_(mx[0] is mx1)
    assert_(mx[1] is not mx2)
    assert_(np.all(mx[1].data == mx2.data))
    assert_(np.all(mx[1].mask))
    mx[1].data[0] = 0.0
    assert_(mx2[0] == 0.0)