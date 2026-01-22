import pytest
import numpy.polynomial as poly
from numpy.core import array
from numpy.testing import assert_equal, assert_raises, assert_
@pytest.mark.parametrize('f', ops)
def test_binops_fails(self, f):
    assert_raises(ValueError, f, self.other)