import random
import functools
import numpy as np
from numpy import array, identity, dot, sqrt
from numpy.testing import (assert_array_almost_equal, assert_allclose, assert_,
import pytest
import scipy.linalg
from scipy.linalg import (funm, signm, logm, sqrtm, fractional_matrix_power,
from scipy.linalg import _matfuncs_inv_ssq
import scipy.linalg._expm_frechet
from scipy.optimize import minimize
def test_defective2(self):
    a = array(([29.2, -24.2, 69.5, 49.8, 7.0], [-9.2, 5.2, -18.0, -16.8, -2.0], [-10.0, 6.0, -20.0, -18.0, -2.0], [-9.6, 9.6, -25.5, -15.4, -2.0], [9.8, -4.8, 18.0, 18.2, 2.0]))
    signm(a, disp=False)