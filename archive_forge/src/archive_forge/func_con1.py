import math
import numpy as np
from numpy.testing import assert_allclose, assert_, assert_array_equal
from scipy.optimize import fmin_cobyla, minimize, Bounds
def con1(self, x):
    return x[0] ** 2 + x[1] ** 2 - 25