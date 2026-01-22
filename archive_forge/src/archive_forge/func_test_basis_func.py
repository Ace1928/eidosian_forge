from math import nan, inf
import pytest
from numpy.core import array, arange, printoptions
import numpy.polynomial as poly
from numpy.testing import assert_equal, assert_
from fractions import Fraction
from decimal import Decimal
def test_basis_func(self):
    p = poly.Chebyshev([1, 2, 3])
    assert_equal(self.as_latex(p), '$x \\mapsto 1.0\\,{T}_{0}(x) + 2.0\\,{T}_{1}(x) + 3.0\\,{T}_{2}(x)$')
    p = poly.Chebyshev([1, 2, 3], domain=[-1, 0])
    assert_equal(self.as_latex(p), '$x \\mapsto 1.0\\,{T}_{0}(1.0 + 2.0x) + 2.0\\,{T}_{1}(1.0 + 2.0x) + 3.0\\,{T}_{2}(1.0 + 2.0x)$')