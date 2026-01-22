import numpy as np
import pytest
import cvxpy
import cvxpy.error as error
import cvxpy.reductions.dgp2dcp.canonicalizers as dgp_atom_canon
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.reductions import solution
from cvxpy.settings import SOLVER_ERROR
from cvxpy.tests.base_test import BaseTest
def test_basic_equality_constraint(self) -> None:
    x = cvxpy.Variable(pos=True)
    dgp = cvxpy.Problem(cvxpy.Minimize(x), [x == 1.0])
    dgp2dcp = cvxpy.reductions.Dgp2Dcp(dgp)
    dcp = dgp2dcp.reduce()
    self.assertIsInstance(dcp.objective.expr, cvxpy.Variable)
    opt = dcp.solve(SOLVER)
    self.assertAlmostEqual(opt, 0.0)
    self.assertAlmostEqual(dcp.variables()[0].value, 0.0)
    dgp.unpack(dgp2dcp.retrieve(dcp.solution))
    self.assertAlmostEqual(dgp.value, 1.0)
    self.assertAlmostEqual(x.value, 1.0)
    dgp._clear_solution()
    dgp.solve(SOLVER, gp=True)
    self.assertAlmostEqual(dgp.value, 1.0)
    self.assertAlmostEqual(x.value, 1.0)