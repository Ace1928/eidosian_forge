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
def test_non_finite_behavior_exact_x(self):
    x = [1, 2, 2.5, 3, 4]
    xp = [1, 2, 3, 4]
    fp = [1, 2, np.inf, 4]
    assert_almost_equal(np.interp(x, xp, fp), [1, 2, np.inf, np.inf, 4])
    fp = [1, 2, np.nan, 4]
    assert_almost_equal(np.interp(x, xp, fp), [1, 2, np.nan, np.nan, 4])