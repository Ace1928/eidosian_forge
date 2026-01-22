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
def test_msg_arg(self):
    """
            Test setting the msg arg to True does not interfere with solve
            """
    prob = LpProblem('test_msg_arg', const.LpMinimize)
    x = LpVariable('x', 0, 4)
    y = LpVariable('y', -1, 1)
    z = LpVariable('z', 0)
    w = LpVariable('w', 0)
    prob += (x + 4 * y + 9 * z, 'obj')
    prob += (x + y <= 5, 'c1')
    prob += (x + z >= 10, 'c2')
    prob += (-y + z == 7, 'c3')
    prob += (w >= 0, 'c4')
    data = prob.toDict()
    var1, prob1 = LpProblem.fromDict(data)
    x, y, z, w = (var1[name] for name in ['x', 'y', 'z', 'w'])
    if self.solver.name in ['HiGHS']:
        return
    self.solver.msg = True
    pulpTestCheck(prob1, self.solver, [const.LpStatusOptimal], {x: 4, y: -1, z: 6, w: 0})