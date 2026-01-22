from math import nan, inf
import pytest
from numpy.core import array, arange, printoptions
import numpy.polynomial as poly
from numpy.testing import assert_equal, assert_
from fractions import Fraction
from decimal import Decimal
def test_simple_polynomial(self):
    p = poly.Polynomial([1, 2, 3])
    assert_equal(self.as_latex(p), '$x \\mapsto 1.0 + 2.0\\,x + 3.0\\,x^{2}$')
    p = poly.Polynomial([1, 2, 3], domain=[-2, 0])
    assert_equal(self.as_latex(p), '$x \\mapsto 1.0 + 2.0\\,\\left(1.0 + x\\right) + 3.0\\,\\left(1.0 + x\\right)^{2}$')
    p = poly.Polynomial([1, 2, 3], domain=[-0.5, 0.5])
    assert_equal(self.as_latex(p), '$x \\mapsto 1.0 + 2.0\\,\\left(2.0x\\right) + 3.0\\,\\left(2.0x\\right)^{2}$')
    p = poly.Polynomial([1, 2, 3], domain=[-1, 0])
    assert_equal(self.as_latex(p), '$x \\mapsto 1.0 + 2.0\\,\\left(1.0 + 2.0x\\right) + 3.0\\,\\left(1.0 + 2.0x\\right)^{2}$')