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
def test_keywords3_ticket_2100(self):

    def mypolyval(x, p):
        _p = list(p)
        res = _p.pop(0)
        while _p:
            res = res * x + _p.pop(0)
        return res
    vpolyval = np.vectorize(mypolyval, excluded=['p', 1])
    ans = [3, 6]
    assert_array_equal(ans, vpolyval(x=[0, 1], p=[1, 2, 3]))
    assert_array_equal(ans, vpolyval([0, 1], p=[1, 2, 3]))
    assert_array_equal(ans, vpolyval([0, 1], [1, 2, 3]))