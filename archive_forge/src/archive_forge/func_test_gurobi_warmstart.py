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
def test_gurobi_warmstart(self) -> None:
    """Test Gurobi warm start with a user provided point.
        """
    if cp.GUROBI in INSTALLED_SOLVERS:
        import gurobipy
        m = 4
        n = 3
        y = Variable(nonneg=True)
        X = Variable((m, n))
        X_vals = np.reshape(np.arange(m * n), (m, n))
        prob = Problem(Minimize(y ** 2 + cp.sum(X)), [X == X_vals])
        X.value = X_vals + 1
        prob.solve(solver=cp.GUROBI, warm_start=True)
        model = prob.solver_stats.extra_stats
        model_x = model.getVars()
        assert gurobipy.GRB.UNDEFINED == model_x[0].start
        assert np.isclose(0, model_x[0].x)
        for i in range(1, X.size + 1):
            row = (i - 1) % X.shape[0]
            col = (i - 1) // X.shape[0]
            assert X_vals[row, col] + 1 == model_x[i].start
            assert np.isclose(X.value[row, col], model_x[i].x)