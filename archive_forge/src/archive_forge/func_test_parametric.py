import numpy as np
import scipy.sparse as sp
from scipy.linalg import lstsq
import cvxpy as cp
from cvxpy import Maximize, Minimize, Parameter, Problem
from cvxpy.atoms import (
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS, QP_SOLVERS
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.solver_test_helpers import StandardTestLPs
def test_parametric(self) -> None:
    """Test solve parametric problem vs full problem"""
    x = Variable()
    a = 10
    b_vec = [-10, -2.0]
    for solver in self.solvers:
        print(solver)
        x_full = []
        obj_full = []
        for b in b_vec:
            obj = Minimize(a * x ** 2 + b * x)
            constraints = [0 <= x, x <= 1]
            prob = Problem(obj, constraints)
            prob.solve(solver=solver)
            x_full += [x.value]
            obj_full += [prob.value]
        x_param = []
        obj_param = []
        b = Parameter()
        obj = Minimize(a * x ** 2 + b * x)
        constraints = [0 <= x, x <= 1]
        prob = Problem(obj, constraints)
        for b_value in b_vec:
            b.value = b_value
            prob.solve(solver=solver)
            x_param += [x.value]
            obj_param += [prob.value]
        print(x_full)
        print(x_param)
        for i in range(len(b_vec)):
            self.assertItemsAlmostEqual(x_full[i], x_param[i], places=3)
            self.assertAlmostEqual(obj_full[i], obj_param[i])