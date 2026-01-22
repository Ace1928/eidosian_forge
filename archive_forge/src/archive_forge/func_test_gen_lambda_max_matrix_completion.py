import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.reductions.dqcp2dcp.dqcp2dcp import Dqcp2Dcp
from cvxpy.reductions.solvers import bisection
from cvxpy.tests import base_test
def test_gen_lambda_max_matrix_completion(self) -> None:
    A = cp.Variable((3, 3))
    B = cp.Variable((3, 3), PSD=True)
    gen_lambda_max = cp.gen_lambda_max(A, B)
    known_indices = tuple(zip(*[[0, 0], [0, 2], [1, 1]]))
    constr = [A[known_indices] == [1.0, 1.9, 0.8], B[known_indices] == [3.0, 1.4, 0.2]]
    problem = cp.Problem(cp.Minimize(gen_lambda_max), constr)
    self.assertTrue(problem.is_dqcp())
    problem.solve(cp.SCS, qcp=True)