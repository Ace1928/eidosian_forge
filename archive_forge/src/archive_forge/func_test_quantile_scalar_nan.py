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
def test_quantile_scalar_nan(self):
    a = np.array([[10.0, 7.0, 4.0], [3.0, 2.0, 1.0]])
    a[0][1] = np.nan
    actual = np.quantile(a, 0.5)
    assert np.isscalar(actual)
    assert_equal(np.quantile(a, 0.5), np.nan)