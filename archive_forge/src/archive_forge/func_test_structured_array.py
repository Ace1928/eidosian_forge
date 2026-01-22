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
def test_structured_array(self):
    a = np.array([(1, 'a'), (2, 'b'), (3, 'c')], dtype=[('foo', 'i'), ('bar', 'a1')])
    val = (4, 'd')
    b = np.insert(a, 0, val)
    assert_array_equal(b[0], np.array(val, dtype=b.dtype))
    val = [(4, 'd')] * 2
    b = np.insert(a, [0, 2], val)
    assert_array_equal(b[[0, 3]], np.array(val, dtype=b.dtype))