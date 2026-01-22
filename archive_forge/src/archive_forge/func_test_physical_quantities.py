import pytest
from numpy import (
from numpy.testing import (
def test_physical_quantities(self):
    a = PhysicalQuantity(0.0)
    b = PhysicalQuantity(1.0)
    assert_equal(linspace(a, b), linspace(0.0, 1.0))