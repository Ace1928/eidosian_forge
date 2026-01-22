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
def test_masked_where_shape_constraint(self):
    a = arange(10)
    with assert_raises(IndexError):
        masked_equal(1, a)
    test = masked_equal(a, 1)
    assert_equal(test.mask, [0, 1, 0, 0, 0, 0, 0, 0, 0, 0])