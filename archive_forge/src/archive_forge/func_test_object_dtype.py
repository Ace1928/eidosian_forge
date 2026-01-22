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
def test_object_dtype(self):
    a = np.array([decimal.Decimal(x) for x in range(10)])
    w = np.array([decimal.Decimal(1) for _ in range(10)])
    w /= w.sum()
    assert_almost_equal(a.mean(0), average(a, weights=w))