import numpy as np
import pytest
import cvxpy
import cvxpy.error as error
import cvxpy.reductions.dgp2dcp.canonicalizers as dgp_atom_canon
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.reductions import solution
from cvxpy.settings import SOLVER_ERROR
from cvxpy.tests.base_test import BaseTest
def test_solver_error(self) -> None:
    x = cvxpy.Variable(pos=True)
    y = cvxpy.Variable(pos=True)
    prod = x * y
    dgp = cvxpy.Problem(cvxpy.Minimize(prod), [])
    dgp2dcp = cvxpy.reductions.Dgp2Dcp()
    _, inverse_data = dgp2dcp.apply(dgp)
    soln = solution.Solution(SOLVER_ERROR, None, {}, {}, {})
    dgp_soln = dgp2dcp.invert(soln, inverse_data)
    self.assertEqual(dgp_soln.status, SOLVER_ERROR)