from numpy.testing import (assert_allclose,
import pytest
import numpy as np
from scipy.optimize import direct, Bounds
def circle_with_args(self, x, a, b):
    return np.square(x[0] - a) + np.square(x[1] - b).sum()