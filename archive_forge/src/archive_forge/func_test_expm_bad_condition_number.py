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
def test_expm_bad_condition_number(self):
    A = np.array([[-1.12867982, 96141.83771, -4524855739.0, 292496941100000.0], [0, -1.201010529, 96346.96872, -4681048289.0], [0, 0, -1.132893222, 95324.9183], [0, 0, 0, -1.179475332]])
    kappa = expm_cond(A)
    assert_array_less(1e+36, kappa)