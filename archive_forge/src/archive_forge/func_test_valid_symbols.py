import pytest
import numpy.polynomial as poly
from numpy.core import array
from numpy.testing import assert_equal, assert_raises, assert_
@pytest.mark.parametrize('symbol', ('x', 'x_1', 'A', 'xyz', 'β'))
def test_valid_symbols(self, symbol):
    """
        Values for symbol that should pass input validation.
        """
    p = poly.Polynomial(self.c, symbol=symbol)
    assert_equal(p.symbol, symbol)