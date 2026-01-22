import numpy as np
import cvxpy as cp
from cvxpy.error import SolverError
from cvxpy.tests.base_test import BaseTest
def test_invalid_variable(self) -> None:
    x = cp.Variable(shape=(2, 2), symmetric=True)
    with self.assertRaises(ValueError):
        cp.suppfunc(x, [])