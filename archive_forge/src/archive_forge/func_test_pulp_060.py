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
def test_pulp_060(self):
    prob = LpProblem('test060', const.LpMinimize)
    x = LpVariable('x', 0, 4, const.LpInteger)
    y = LpVariable('y', -1, 1, const.LpInteger)
    z = LpVariable('z', 0, 10, const.LpInteger)
    prob += (x + y <= 5.2, 'c1')
    prob += (x + z >= 10.3, 'c2')
    prob += (-y + z == 7.4, 'c3')
    print('\t Testing an integer infeasible problem')
    if self.solver.__class__ in [GLPK_CMD, COIN_CMD, PULP_CBC_CMD, MOSEK]:
        pulpTestCheck(prob, self.solver, [const.LpStatusInfeasible, const.LpStatusUndefined])
    elif self.solver.__class__ in [COINMP_DLL]:
        print('\t\t Error in CoinMP to be fixed, reports Optimal')
        pulpTestCheck(prob, self.solver, [const.LpStatusOptimal])
    elif self.solver.__class__ in [GUROBI_CMD, FSCIP_CMD]:
        pulpTestCheck(prob, self.solver, [const.LpStatusNotSolved])
    else:
        pulpTestCheck(prob, self.solver, [const.LpStatusInfeasible])