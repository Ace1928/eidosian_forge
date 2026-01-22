import os
import sys
import time
from itertools import zip_longest
import numpy as np
from numpy.testing import assert_
import pytest
from scipy.special._testutils import assert_func_equal
def assert_mpmath_equal(*a, **kw):
    d = MpmathData(*a, **kw)
    d.check()