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
def test_pulp_070(self):
    prob = LpProblem('test070', const.LpMinimize)
    obj = LpConstraintVar('obj')
    a = LpConstraintVar('C1', const.LpConstraintLE, 5)
    b = LpConstraintVar('C2', const.LpConstraintGE, 10)
    c = LpConstraintVar('C3', const.LpConstraintEQ, 7)
    prob.setObjective(obj)
    prob += a
    prob += b
    prob += c
    x = LpVariable('x', 0, 4, const.LpContinuous, obj + a + b)
    y = LpVariable('y', -1, 1, const.LpContinuous, 4 * obj + a - c)
    z = LpVariable('z', 0, None, const.LpContinuous, 9 * obj + b + c)
    print('\t Testing column based modelling')
    pulpTestCheck(prob, self.solver, [const.LpStatusOptimal], {x: 4, y: -1, z: 6})