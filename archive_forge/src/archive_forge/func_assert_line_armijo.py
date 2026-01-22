from numpy.testing import (assert_equal, assert_array_almost_equal,
import scipy.optimize._linesearch as ls
from scipy.optimize._linesearch import LineSearchWarning
import numpy as np
def assert_line_armijo(x, p, s, f, **kw):
    assert_armijo(s, phi=lambda sp: f(x + p * sp), **kw)