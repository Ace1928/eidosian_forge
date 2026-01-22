from math import nan, inf
import pytest
from numpy.core import array, arange, printoptions
import numpy.polynomial as poly
from numpy.testing import assert_equal, assert_
from fractions import Fraction
from decimal import Decimal
def test_laguerre_repr(self):
    res = repr(poly.Laguerre([0, 1]))
    tgt = "Laguerre([0., 1.], domain=[0, 1], window=[0, 1], symbol='x')"
    assert_equal(res, tgt)