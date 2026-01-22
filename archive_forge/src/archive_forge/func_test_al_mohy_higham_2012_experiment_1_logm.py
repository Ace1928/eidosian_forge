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
def test_al_mohy_higham_2012_experiment_1_logm(self):
    A = _get_al_mohy_higham_2012_experiment_1()
    A_logm, info = logm(A, disp=False)
    A_round_trip = expm(A_logm)
    assert_allclose(A_round_trip, A, rtol=5e-05, atol=1e-14)