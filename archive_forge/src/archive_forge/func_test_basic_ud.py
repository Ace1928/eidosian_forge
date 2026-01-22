import operator
import warnings
import sys
import decimal
from fractions import Fraction
import math
import pytest
import hypothesis
from hypothesis.extra.numpy import arrays
import hypothesis.strategies as st
from functools import partial
import numpy as np
from numpy import ma
from numpy.testing import (
import numpy.lib.function_base as nfb
from numpy.random import rand
from numpy.lib import (
from numpy.core.numeric import normalize_axis_tuple
def test_basic_ud(self):
    a = get_mat(4)
    b = a[::-1, :]
    assert_equal(np.flip(a, 0), b)
    a = [[0, 1, 2], [3, 4, 5]]
    b = [[3, 4, 5], [0, 1, 2]]
    assert_equal(np.flip(a, 0), b)