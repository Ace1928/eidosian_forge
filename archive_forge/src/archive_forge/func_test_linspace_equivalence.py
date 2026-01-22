import pytest
import numpy as np
from numpy.testing import (
from numpy.lib.index_tricks import (
def test_linspace_equivalence(self):
    y, st = np.linspace(2, 10, retstep=True)
    assert_almost_equal(st, 8 / 49.0)
    assert_array_almost_equal(y, mgrid[2:10:50j], 13)