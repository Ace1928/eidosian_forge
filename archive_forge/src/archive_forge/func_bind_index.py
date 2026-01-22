from numpy.testing import (assert_equal, assert_array_almost_equal,
import scipy.optimize._linesearch as ls
from scipy.optimize._linesearch import LineSearchWarning
import numpy as np
def bind_index(func, idx):
    return lambda *a, **kw: func(*a, **kw)[idx]