from numpy.testing import (
from numpy import (
import numpy as np
import pytest
def test_diag2d(self):
    assert_equal(eye(3, 4, k=2), array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]]))
    assert_equal(eye(4, 3, k=-2), array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]]))