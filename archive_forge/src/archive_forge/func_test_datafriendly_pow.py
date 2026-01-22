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
def test_datafriendly_pow(self):
    x = array([1.0, 2.0, 3.0], mask=[0, 0, 1])
    xx = x ** 2.5
    assert_equal(xx.data, [1.0, 2.0 ** 2.5, 3.0])
    assert_equal(xx.mask, [0, 0, 1])
    x **= 2.5
    assert_equal(x.data, [1.0, 2.0 ** 2.5, 3])
    assert_equal(x.mask, [0, 0, 1])