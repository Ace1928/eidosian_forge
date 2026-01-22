from functools import partial
from itertools import product
import numpy as np
import pytest
from numpy.testing import (assert_allclose, assert_, assert_equal,
from scipy.sparse import SparseEfficiencyWarning
from scipy.sparse.linalg import aslinearoperator
import scipy.linalg
from scipy.sparse.linalg import expm as sp_expm
from scipy.sparse.linalg._expm_multiply import (_theta, _compute_p_max,
from scipy._lib._util import np_long
def test_expm_multiply_interval_vector(self):
    np.random.seed(1234)
    interval = {'start': 0.1, 'stop': 3.2, 'endpoint': True}
    for num, n in product([14, 13, 2], [1, 2, 5, 20, 40]):
        A = scipy.linalg.inv(np.random.randn(n, n))
        v = np.random.randn(n)
        samples = np.linspace(num=num, **interval)
        X = expm_multiply(A, v, num=num, **interval)
        for solution, t in zip(X, samples):
            assert_allclose(solution, sp_expm(t * A).dot(v))
        Xguess = estimated(expm_multiply)(aslinearoperator(A), v, num=num, **interval)
        Xgiven = expm_multiply(aslinearoperator(A), v, num=num, **interval, traceA=np.trace(A))
        Xwrong = expm_multiply(aslinearoperator(A), v, num=num, **interval, traceA=np.trace(A) * 5)
        for sol_guess, sol_given, sol_wrong, t in zip(Xguess, Xgiven, Xwrong, samples):
            correct = sp_expm(t * A).dot(v)
            assert_allclose(sol_guess, correct)
            assert_allclose(sol_given, correct)
            assert_allclose(sol_wrong, correct)