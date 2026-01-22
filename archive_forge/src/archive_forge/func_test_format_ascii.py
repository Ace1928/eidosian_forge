from math import nan, inf
import pytest
from numpy.core import array, arange, printoptions
import numpy.polynomial as poly
from numpy.testing import assert_equal, assert_
from fractions import Fraction
from decimal import Decimal
def test_format_ascii(self):
    poly.set_default_printstyle('unicode')
    p = poly.Polynomial([1, 2, 0, -1])
    assert_equal(format(p, 'ascii'), '1.0 + 2.0 x + 0.0 x**2 - 1.0 x**3')