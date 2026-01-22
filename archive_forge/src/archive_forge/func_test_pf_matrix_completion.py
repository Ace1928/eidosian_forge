import numpy as np
import pytest
import cvxpy
import cvxpy.error as error
import cvxpy.reductions.dgp2dcp.canonicalizers as dgp_atom_canon
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.reductions import solution
from cvxpy.settings import SOLVER_ERROR
from cvxpy.tests.base_test import BaseTest
def test_pf_matrix_completion(self) -> None:
    X = cvxpy.Variable((3, 3), pos=True)
    obj = cvxpy.Minimize(cvxpy.pf_eigenvalue(X))
    known_indices = tuple(zip(*[[0, 0], [0, 2], [1, 1], [2, 0], [2, 1]]))
    constr = [X[known_indices] == [1.0, 1.9, 0.8, 3.2, 5.9], X[0, 1] * X[1, 0] * X[1, 2] * X[2, 2] == 1.0]
    problem = cvxpy.Problem(obj, constr)
    problem.solve(SOLVER, gp=True)