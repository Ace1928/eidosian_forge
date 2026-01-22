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
def test_right_basic(self):
    x = [1, 5, 4, 10, 8, 11, 0]
    bins = [1, 5, 10]
    default_answer = [1, 2, 1, 3, 2, 3, 0]
    assert_array_equal(digitize(x, bins), default_answer)
    right_answer = [0, 1, 1, 2, 2, 3, 0]
    assert_array_equal(digitize(x, bins, True), right_answer)