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
def test_masked_singleton_equality(self):
    a = array([1, 2, 3], mask=[1, 1, 0])
    assert_((a[0] == 0) is masked)
    assert_((a[0] != 0) is masked)
    assert_equal(a[-1] == 0, False)
    assert_equal(a[-1] != 0, True)