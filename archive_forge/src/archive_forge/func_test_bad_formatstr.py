from math import nan, inf
import pytest
from numpy.core import array, arange, printoptions
import numpy.polynomial as poly
from numpy.testing import assert_equal, assert_
from fractions import Fraction
from decimal import Decimal
def test_bad_formatstr(self):
    p = poly.Polynomial([1, 2, 0, -1])
    with pytest.raises(ValueError):
        format(p, '.2f')