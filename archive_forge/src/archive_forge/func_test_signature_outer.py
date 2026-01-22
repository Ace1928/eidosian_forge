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
def test_signature_outer(self):
    f = vectorize(np.outer, signature='(a),(b)->(a,b)')
    r = f([1, 2], [1, 2, 3])
    assert_array_equal(r, [[1, 2, 3], [2, 4, 6]])
    r = f([[[1, 2]]], [1, 2, 3])
    assert_array_equal(r, [[[[1, 2, 3], [2, 4, 6]]]])
    r = f([[1, 0], [2, 0]], [1, 2, 3])
    assert_array_equal(r, [[[1, 2, 3], [0, 0, 0]], [[2, 4, 6], [0, 0, 0]]])
    r = f([1, 2], [[1, 2, 3], [0, 0, 0]])
    assert_array_equal(r, [[[1, 2, 3], [2, 4, 6]], [[0, 0, 0], [0, 0, 0]]])