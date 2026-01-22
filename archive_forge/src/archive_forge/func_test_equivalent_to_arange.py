import pytest
from numpy import (
from numpy.testing import (
def test_equivalent_to_arange(self):
    for j in range(1000):
        assert_equal(linspace(0, j, j + 1, dtype=int), arange(j + 1, dtype=int))