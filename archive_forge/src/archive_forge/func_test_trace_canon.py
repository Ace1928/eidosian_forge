import numpy as np
import pytest
import cvxpy
import cvxpy.error as error
import cvxpy.reductions.dgp2dcp.canonicalizers as dgp_atom_canon
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.reductions import solution
from cvxpy.settings import SOLVER_ERROR
from cvxpy.tests.base_test import BaseTest
def test_trace_canon(self) -> None:
    X = cvxpy.Constant(np.array([[1.0, 5.0], [9.0, 14.0]]))
    Y = cvxpy.trace(X)
    canon, constraints = dgp_atom_canon.trace_canon(Y, Y.args)
    self.assertEqual(len(constraints), 0)
    self.assertTrue(canon.is_scalar())
    expected = np.log(np.exp(1.0) + np.exp(14.0))
    self.assertAlmostEqual(expected, canon.value)