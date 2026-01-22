import pytest
from numpy.testing import assert_allclose, assert_equal
import numpy as np
from math import pow
from scipy import optimize
def fg38(self, x):
    return (self.f38(x), self.g38(x))