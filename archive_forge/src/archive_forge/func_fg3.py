import pytest
from numpy.testing import assert_allclose, assert_equal
import numpy as np
from math import pow
from scipy import optimize
def fg3(self, x):
    return (self.f3(x), self.g3(x))