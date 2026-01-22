import pytest
import numpy.polynomial as poly
from numpy.core import array
from numpy.testing import assert_equal, assert_raises, assert_
def test_integ(self):
    other = self.p.integ()
    assert_equal(other.symbol, 'z')