import os
import tempfile
from pulp.constants import PulpError
from pulp.apis import *
from pulp import LpVariable, LpProblem, lpSum, LpConstraintVar, LpFractionConstraint
from pulp import constants as const
from pulp.tests.bin_packing_problem import create_bin_packing_problem
from pulp.utilities import makeDict
import functools
import unittest
def test_pulp_012(self):
    prob = LpProblem('test012', const.LpMaximize)
    x = LpVariable('x', 0, 4)
    y = LpVariable('y', -1, 1)
    z = LpVariable('z', 0)
    w = LpVariable('w', 0)
    prob += (x + 4 * y + 9 * z + w, 'obj')
    prob += (x + y <= 5, 'c1')
    prob += (x + z >= 10, 'c2')
    prob += (-y + z == 7, 'c3')
    prob += (w >= 0, 'c4')
    print('\t Testing unbounded continuous LP solution')
    if self.solver.__class__ in [GUROBI, CPLEX_CMD, YAPOSIB, MOSEK, COPT]:
        pulpTestCheck(prob, self.solver, [const.LpStatusInfeasible, const.LpStatusUnbounded])
    elif self.solver.__class__ in [COINMP_DLL, MIPCL_CMD]:
        print('\t\t Error in CoinMP and MIPCL_CMD: reports Optimal')
        pulpTestCheck(prob, self.solver, [const.LpStatusOptimal])
    elif self.solver.__class__ is GLPK_CMD:
        pulpTestCheck(prob, self.solver, [const.LpStatusUndefined])
    elif self.solver.__class__ in [GUROBI_CMD, SCIP_CMD, FSCIP_CMD, SCIP_PY]:
        pulpTestCheck(prob, self.solver, [const.LpStatusNotSolved])
    elif self.solver.__class__ in [CHOCO_CMD]:
        pass
    else:
        pulpTestCheck(prob, self.solver, [const.LpStatusUnbounded])