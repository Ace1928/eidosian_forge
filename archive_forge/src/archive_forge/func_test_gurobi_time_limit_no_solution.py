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
def test_gurobi_time_limit_no_solution(self) -> None:
    """Make sure that if Gurobi terminates due to a time limit before finding a solution:
            1) no error is raised,
            2) solver stats are returned.
            The test is skipped if something changes on Gurobi's side so that:
            - a solution is found despite a time limit of zero,
            - a different termination criteria is hit first.
        """
    from cvxpy import GUROBI
    if GUROBI in INSTALLED_SOLVERS:
        import gurobipy
        objective = Minimize(self.x[0])
        constraints = [self.x[0] >= 1]
        prob = Problem(objective, constraints)
        try:
            prob.solve(solver=GUROBI, TimeLimit=0.0)
        except Exception as e:
            self.fail('An exception %s is raised instead of returning a result.' % e)
        extra_stats = None
        solver_stats = getattr(prob, 'solver_stats', None)
        if solver_stats:
            extra_stats = getattr(solver_stats, 'extra_stats', None)
        self.assertTrue(extra_stats, 'Solver stats have not been returned.')
        nb_solutions = getattr(extra_stats, 'SolCount', None)
        if nb_solutions:
            self.skipTest('Gurobi has found a solution, the test is not relevant anymore.')
        solver_status = getattr(extra_stats, 'Status', None)
        if solver_status != gurobipy.StatusConstClass.TIME_LIMIT:
            self.skipTest('Gurobi terminated for a different reason than reaching time limit, the test is not relevant anymore.')
    else:
        with self.assertRaises(Exception) as cm:
            prob = Problem(Minimize(norm(self.x, 1)), [self.x == 0])
            prob.solve(solver=GUROBI, TimeLimit=0)
        self.assertEqual(str(cm.exception), 'The solver %s is not installed.' % GUROBI)