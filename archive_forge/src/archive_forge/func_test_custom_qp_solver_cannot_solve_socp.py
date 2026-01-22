import unittest
import cvxpy as cp
from cvxpy.reductions.solvers.conic_solvers.scs_conif import SCS
from cvxpy.reductions.solvers.qp_solvers.osqp_qpif import OSQP
def test_custom_qp_solver_cannot_solve_socp(self) -> None:
    with self.assertRaises(cp.error.SolverError):
        self.solve_example_socp(solver=self.custom_qp_solver)