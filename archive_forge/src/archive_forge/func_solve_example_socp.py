import unittest
import cvxpy as cp
from cvxpy.reductions.solvers.conic_solvers.scs_conif import SCS
from cvxpy.reductions.solvers.qp_solvers.osqp_qpif import OSQP
@staticmethod
def solve_example_socp(solver) -> None:
    x = cp.Variable(2)
    y = cp.Variable()
    quadratic = cp.sum_squares(x)
    problem = cp.Problem(cp.Minimize(quadratic), [cp.SOC(y, x)])
    problem.solve(solver=solver)