import platform
import warnings
import fnmatch
import itertools
import pytest
import sys
import os
import operator
from fractions import Fraction
from functools import reduce
from collections import namedtuple
import numpy.core.umath as ncu
from numpy.core import _umath_tests as ncu_tests
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _glibc_older_than
@pytest.mark.skipif(np.finfo(np.double) == np.finfo(np.longdouble), reason='long double is same as double')
@pytest.mark.xfail(condition=platform.machine().startswith('ppc64'), reason='IBM double double')
def test_spacingl():
    return _test_spacing(np.longdouble)