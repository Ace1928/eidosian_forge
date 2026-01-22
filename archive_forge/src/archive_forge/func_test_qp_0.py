import warnings
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
@staticmethod
def test_qp_0(solver, places: int=4, duals: bool=True, **kwargs) -> SolverTestHelper:
    sth = qp_0()
    sth.solve(solver, **kwargs)
    sth.verify_primal_values(places)
    sth.verify_objective(places)
    if duals:
        sth.check_complementarity(places)
        sth.verify_dual_values(places)
    return sth