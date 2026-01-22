import warnings
import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.tests.base_test import BaseTest
def test_sum_matrix(self) -> None:
    w = cp.Variable((2, 2), pos=True)
    h = cp.Variable((2, 2), pos=True)
    alpha = cp.Parameter(pos=True, value=1.0)
    beta = cp.Parameter(pos=True, value=20)
    kappa = cp.Parameter(pos=True, value=10)
    problem = cp.Problem(cp.Minimize(alpha * cp.sum(h)), [cp.multiply(w, h) >= beta, cp.sum(w) <= kappa])
    gradcheck(problem, gp=True, solve_methods=[s.SCS], atol=0.1)
    perturbcheck(problem, gp=True, solve_methods=[s.SCS], atol=0.1)