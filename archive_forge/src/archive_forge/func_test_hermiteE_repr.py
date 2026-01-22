from math import nan, inf
import pytest
from numpy.core import array, arange, printoptions
import numpy.polynomial as poly
from numpy.testing import assert_equal, assert_
from fractions import Fraction
from decimal import Decimal
def test_hermiteE_repr(self):
    res = repr(poly.HermiteE([0, 1]))
    tgt = "HermiteE([0., 1.], domain=[-1,  1], window=[-1,  1], symbol='x')"
    assert_equal(res, tgt)