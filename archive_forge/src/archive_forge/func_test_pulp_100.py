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
def test_pulp_100(self):
    """
            Test the ability to sequentially solve a problem
            """
    prob = LpProblem('test100', const.LpMinimize)
    x = LpVariable('x', 0, 1)
    y = LpVariable('y', 0, 1)
    z = LpVariable('z', 0, 1)
    obj1 = x + 0 * y + 0 * z
    obj2 = 0 * x - 1 * y + 0 * z
    prob += (x <= 1, 'c1')
    if self.solver.__class__ in [COINMP_DLL, GUROBI]:
        print('\t Testing Sequential Solves')
        status = prob.sequentialSolve([obj1, obj2], solver=self.solver)
        pulpTestCheck(prob, self.solver, [[const.LpStatusOptimal, const.LpStatusOptimal]], sol={x: 0, y: 1}, status=status)