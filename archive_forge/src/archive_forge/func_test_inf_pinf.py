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
def test_inf_pinf(self):
    assert_almost_equal(ncu.arctan2(np.inf, np.inf), 0.25 * np.pi)
    assert_almost_equal(ncu.arctan2(-np.inf, np.inf), -0.25 * np.pi)