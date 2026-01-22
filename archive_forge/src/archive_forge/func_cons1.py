import math
import numpy as np
from numpy.testing import assert_allclose, assert_, assert_array_equal
from scipy.optimize import fmin_cobyla, minimize, Bounds
def cons1(x):
    a = np.array([[1, -2, 2], [-1, -2, 6], [-1, 2, 2]])
    return np.array([a[i, 0] * x[0] + a[i, 1] * x[1] + a[i, 2] for i in range(len(a))])