import numpy as np
import pytest
import cvxpy
import cvxpy.error as error
import cvxpy.reductions.dgp2dcp.canonicalizers as dgp_atom_canon
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.reductions import solution
from cvxpy.settings import SOLVER_ERROR
from cvxpy.tests.base_test import BaseTest
def test_parameter_name(self) -> None:
    param = cvxpy.Parameter(pos=True, name='alpha')
    param.value = 1.0
    dgp = cvxpy.Problem(cvxpy.Minimize(param), [])
    dgp2dcp = cvxpy.reductions.Dgp2Dcp(dgp)
    dcp = dgp2dcp.reduce()
    self.assertAlmostEqual(dcp.parameters()[0].name(), 'alpha')