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
def test_nextafter_vs_spacing():
    for t in [np.float32, np.float64]:
        for _f in [1, 1e-05, 1000]:
            f = t(_f)
            f1 = t(_f + 1)
            assert_(np.nextafter(f, f1) - f == np.spacing(f))