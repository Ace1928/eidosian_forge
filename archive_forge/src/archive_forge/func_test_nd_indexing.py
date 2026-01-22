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
def test_nd_indexing(self):
    a, b, c = np.meshgrid([0], [1, 2], [3, 4, 5], indexing='ij')
    assert_equal(a, [[[0, 0, 0], [0, 0, 0]]])
    assert_equal(b, [[[1, 1, 1], [2, 2, 2]]])
    assert_equal(c, [[[3, 4, 5], [3, 4, 5]]])