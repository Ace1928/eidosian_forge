import unittest
import cvxpy as cp
from cvxpy.reductions.solvers.conic_solvers.scs_conif import SCS
from cvxpy.reductions.solvers.qp_solvers.osqp_qpif import OSQP
@staticmethod
def solve_example_qp(solver) -> None:
    x = cp.Variable()
    quadratic = cp.sum_squares(x)
    problem = cp.Problem(cp.Minimize(quadratic))
    problem.solve(solver=solver)