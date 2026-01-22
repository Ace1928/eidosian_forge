import warnings
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
@staticmethod
def test_mi_pcp_0(solver, places: int=3, **kwargs) -> SolverTestHelper:
    sth = mi_pcp_0()
    sth.solve(solver, **kwargs)
    sth.verify_objective(places)
    sth.check_primal_feasibility(places)
    sth.verify_primal_values(places)
    return sth