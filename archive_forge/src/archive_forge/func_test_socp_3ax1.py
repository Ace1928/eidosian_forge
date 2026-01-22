import warnings
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
@staticmethod
def test_socp_3ax1(solver, places: int=3, duals: bool=True, **kwargs) -> SolverTestHelper:
    sth = socp_3(axis=1)
    sth.solve(solver, **kwargs)
    sth.verify_objective(places)
    sth.verify_primal_values(places)
    if duals:
        sth.check_complementarity(places)
        sth.verify_dual_values(places)
    return sth