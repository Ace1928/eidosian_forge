from math import nan, inf
import pytest
from numpy.core import array, arange, printoptions
import numpy.polynomial as poly
from numpy.testing import assert_equal, assert_
from fractions import Fraction
from decimal import Decimal
def test_multichar_basis_func(self):
    p = poly.HermiteE([1, 2, 3])
    assert_equal(self.as_latex(p), '$x \\mapsto 1.0\\,{He}_{0}(x) + 2.0\\,{He}_{1}(x) + 3.0\\,{He}_{2}(x)$')