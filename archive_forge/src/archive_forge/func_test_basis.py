import pytest
import numpy.polynomial as poly
from numpy.core import array
from numpy.testing import assert_equal, assert_raises, assert_
def test_basis():
    p = poly.Polynomial.basis(3, symbol='z')
    assert_equal(p.symbol, 'z')