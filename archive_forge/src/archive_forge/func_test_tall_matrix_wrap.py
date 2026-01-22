import pytest
import numpy as np
from numpy.testing import (
from numpy.lib.index_tricks import (
def test_tall_matrix_wrap(self):
    a = np.zeros((10, 3), int)
    fill_diagonal(a, 5, True)
    assert_array_equal(a, np.array([[5, 0, 0], [0, 5, 0], [0, 0, 5], [0, 0, 0], [5, 0, 0], [0, 5, 0], [0, 0, 5], [0, 0, 0], [5, 0, 0], [0, 5, 0]]))