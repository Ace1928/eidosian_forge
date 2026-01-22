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
def test_keepmask(self):
    x = masked_array([1, 2, 3], mask=[1, 0, 0])
    mx = masked_array(x)
    assert_equal(mx.mask, x.mask)
    mx = masked_array(x, mask=[0, 1, 0], keep_mask=False)
    assert_equal(mx.mask, [0, 1, 0])
    mx = masked_array(x, mask=[0, 1, 0], keep_mask=True)
    assert_equal(mx.mask, [1, 1, 0])
    mx = masked_array(x, mask=[0, 1, 0])
    assert_equal(mx.mask, [1, 1, 0])