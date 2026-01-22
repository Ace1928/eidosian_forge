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
def test_logPath(self):
    name = self._testMethodName
    prob = LpProblem(name, const.LpMinimize)
    x = LpVariable('x', 0, 4)
    y = LpVariable('y', -1, 1)
    z = LpVariable('z', 0)
    w = LpVariable('w', 0)
    prob += (x + 4 * y + 9 * z, 'obj')
    prob += (x + y <= 5, 'c1')
    prob += (x + z >= 10, 'c2')
    prob += (-y + z == 7, 'c3')
    prob += (w >= 0, 'c4')
    logFilename = name + '.log'
    self.solver.optionsDict['logPath'] = logFilename
    if self.solver.name in ['CPLEX_PY', 'CPLEX_CMD', 'GUROBI', 'GUROBI_CMD', 'PULP_CBC_CMD', 'COIN_CMD']:
        print('\t Testing logPath argument')
        pulpTestCheck(prob, self.solver, [const.LpStatusOptimal], {x: 4, y: -1, z: 6, w: 0})
        if not os.path.exists(logFilename):
            raise PulpError(f'Test failed for solver: {self.solver}')
        if not os.path.getsize(logFilename):
            raise PulpError(f'Test failed for solver: {self.solver}')