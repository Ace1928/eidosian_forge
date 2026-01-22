import unittest
import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.reductions.solvers.defines import INSTALLED_MI_SOLVERS as MIP_SOLVERS
from cvxpy.tests.base_test import BaseTest
def test_all_solvers(self) -> None:
    for solver in self.solvers:
        self.bool_prob(solver)
        if solver != cp.SCIPY:
            self.int_prob(solver)
        if solver in [cp.CPLEX, cp.GUROBI, cp.MOSEK, cp.XPRESS]:
            if solver != cp.XPRESS:
                self.bool_socp(solver)
            self.int_socp(solver)