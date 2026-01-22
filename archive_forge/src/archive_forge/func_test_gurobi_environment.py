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
def test_gurobi_environment(self) -> None:
    """Tests that Gurobi environments can be passed to Model.
        Gurobi environments can include licensing and model parameter data.
        """
    from cvxpy import GUROBI
    if GUROBI in INSTALLED_SOLVERS:
        import gurobipy
        params = {'MIPGap': np.random.random(), 'AggFill': np.random.randint(10), 'PerturbValue': np.random.random()}
        custom_env = gurobipy.Env()
        for k, v in params.items():
            custom_env.setParam(k, v)
        sth = StandardTestLPs.test_lp_0(solver='GUROBI', env=custom_env)
        model = sth.prob.solver_stats.extra_stats
        for k, v in params.items():
            name, p_type, p_val, p_min, p_max, p_def = model.getParamInfo(k)
            self.assertEqual(v, p_val)
    else:
        with self.assertRaises(Exception) as cm:
            prob = Problem(Minimize(norm(self.x, 1)), [self.x == 0])
            prob.solve(solver=GUROBI, TimeLimit=0)
        self.assertEqual(str(cm.exception), 'The solver %s is not installed.' % GUROBI)