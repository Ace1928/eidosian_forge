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
def test_pulp_050(self):
    prob = LpProblem('test050', const.LpMinimize)
    x = LpVariable('x', 0, 4)
    y = LpVariable('y', -1, 1)
    z = LpVariable('z', 0, 10)
    prob += (x + y <= 5.2, 'c1')
    prob += (x + z >= 10.3, 'c2')
    prob += (-y + z == 17.5, 'c3')
    print('\t Testing an infeasible problem')
    if self.solver.__class__ is GLPK_CMD:
        pulpTestCheck(prob, self.solver, [const.LpStatusUndefined])
    elif self.solver.__class__ in [GUROBI_CMD, FSCIP_CMD]:
        pulpTestCheck(prob, self.solver, [const.LpStatusNotSolved])
    else:
        pulpTestCheck(prob, self.solver, [const.LpStatusInfeasible])