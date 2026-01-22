import pytest
from numpy.testing import assert_allclose, assert_equal
import numpy as np
from math import pow
from scipy import optimize
def g45(self, x):
    dif = [0] * 5
    dif[0] = -x[1] * x[2] * x[3] * x[4] / 120.0
    dif[1] = -x[0] * x[2] * x[3] * x[4] / 120.0
    dif[2] = -x[0] * x[1] * x[3] * x[4] / 120.0
    dif[3] = -x[0] * x[1] * x[2] * x[4] / 120.0
    dif[4] = -x[0] * x[1] * x[2] * x[3] / 120.0
    return dif