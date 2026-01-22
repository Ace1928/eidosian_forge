import pytest
from numpy.testing import assert_allclose, assert_equal
import numpy as np
from math import pow
from scipy import optimize
def f45(self, x):
    return 2.0 - x[0] * x[1] * x[2] * x[3] * x[4] / 120.0