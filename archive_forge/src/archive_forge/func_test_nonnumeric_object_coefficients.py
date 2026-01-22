from math import nan, inf
import pytest
from numpy.core import array, arange, printoptions
import numpy.polynomial as poly
from numpy.testing import assert_equal, assert_
from fractions import Fraction
from decimal import Decimal
@pytest.mark.parametrize(('coefs', 'tgt'), ((array([1, 2, 'f'], dtype=object), '1 + 2·x + f·x²'), (array([1, 2, [3, 4]], dtype=object), '1 + 2·x + [3, 4]·x²')))
def test_nonnumeric_object_coefficients(coefs, tgt):
    """
    Test coef fallback for object arrays of non-numeric coefficients.
    """
    p = poly.Polynomial(coefs)
    poly.set_default_printstyle('unicode')
    assert_equal(str(p), tgt)