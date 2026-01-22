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
def test_rotation_axes(self):
    a = np.arange(8).reshape((2, 2, 2))
    a_rot90_01 = [[[2, 3], [6, 7]], [[0, 1], [4, 5]]]
    a_rot90_12 = [[[1, 3], [0, 2]], [[5, 7], [4, 6]]]
    a_rot90_20 = [[[4, 0], [6, 2]], [[5, 1], [7, 3]]]
    a_rot90_10 = [[[4, 5], [0, 1]], [[6, 7], [2, 3]]]
    assert_equal(rot90(a, axes=(0, 1)), a_rot90_01)
    assert_equal(rot90(a, axes=(1, 0)), a_rot90_10)
    assert_equal(rot90(a, axes=(1, 2)), a_rot90_12)
    for k in range(1, 5):
        assert_equal(rot90(a, k=k, axes=(2, 0)), rot90(a_rot90_20, k=k - 1, axes=(2, 0)))