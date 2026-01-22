import itertools
import numpy as np
from numpy import exp
from numpy.testing import assert_, assert_equal
from scipy.optimize import root
def test_linear_definite():

    def check_solvability(A, b, line_search='cruz'):

        def func(x):
            return A.dot(x) - b
        xp = np.linalg.solve(A, b)
        eps = np.linalg.norm(func(xp)) * 1000.0
        sol = root(func, b, options=dict(fatol=eps, ftol=0, maxfev=17523, line_search=line_search), method='DF-SANE')
        assert_(sol.success)
        assert_(np.linalg.norm(func(sol.x)) <= eps)
    n = 90
    np.random.seed(1234)
    A = np.arange(n * n).reshape(n, n)
    A = A + n * n * np.diag(1 + np.arange(n))
    assert_(np.linalg.eigvals(A).min() > 0)
    b = np.arange(n) * 1.0
    check_solvability(A, b, 'cruz')
    check_solvability(A, b, 'cheng')
    check_solvability(-A, b, 'cruz')
    check_solvability(-A, b, 'cheng')