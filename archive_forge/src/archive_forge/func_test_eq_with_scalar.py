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
def test_eq_with_scalar(self):
    a = array(1)
    assert_equal(a == 1, True)
    assert_equal(a == 0, False)
    assert_equal(a != 1, False)
    assert_equal(a != 0, True)
    b = array(1, mask=True)
    assert_equal(b == 0, masked)
    assert_equal(b == 1, masked)
    assert_equal(b != 0, masked)
    assert_equal(b != 1, masked)