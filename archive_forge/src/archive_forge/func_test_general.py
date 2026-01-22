import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
def test_general(self):
    x = np.array([-1, 0, 0.5, 2, 4, 4.5, 5.5, 9])
    y = np.array([0, -0.5, 2, 3, 2.5, 1, 1, 0.5])
    for n in [2, 3, x.size]:
        self.check_all_bc(x[:n], y[:n], 0)
        Y = np.empty((2, n, 2))
        Y[0, :, 0] = y[:n]
        Y[0, :, 1] = y[:n] - 1
        Y[1, :, 0] = y[:n] + 2
        Y[1, :, 1] = y[:n] + 3
        self.check_all_bc(x[:n], Y, 1)