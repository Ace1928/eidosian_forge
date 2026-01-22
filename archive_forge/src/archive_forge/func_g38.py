import pytest
from numpy.testing import assert_allclose, assert_equal
import numpy as np
from math import pow
from scipy import optimize
def g38(self, x):
    dif = [0, 0, 0, 0]
    dif[0] = (-400.0 * x[0] * (x[1] - pow(x[0], 2)) - 2.0 * (1.0 - x[0])) * 1e-05
    dif[1] = (200.0 * (x[1] - pow(x[0], 2)) + 20.2 * (x[1] - 1.0) + 19.8 * (x[3] - 1.0)) * 1e-05
    dif[2] = (-360.0 * x[2] * (x[3] - pow(x[2], 2)) - 2.0 * (1.0 - x[2])) * 1e-05
    dif[3] = (180.0 * (x[3] - pow(x[2], 2)) + 20.2 * (x[3] - 1.0) + 19.8 * (x[1] - 1.0)) * 1e-05
    return dif