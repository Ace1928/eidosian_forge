import numpy as np
import pytest
import cvxpy
import cvxpy.error as error
import cvxpy.reductions.dgp2dcp.canonicalizers as dgp_atom_canon
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.reductions import solution
from cvxpy.settings import SOLVER_ERROR
from cvxpy.tests.base_test import BaseTest
def test_qp_solver_not_allowed(self) -> None:
    x = cvxpy.Variable(pos=True)
    problem = cvxpy.Problem(cvxpy.Minimize(x))
    error_msg = "When `gp=True`, `solver` must be a conic solver (received 'OSQP'); try calling `solve()` with `solver=cvxpy.ECOS`."
    with self.assertRaises(error.SolverError) as err:
        problem.solve(solver='OSQP', gp=True)
        self.assertEqual(error_msg, str(err))