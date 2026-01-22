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
def test_importMPS_noname(self):
    name = self._testMethodName
    prob = LpProblem('', const.LpMaximize)
    x = LpVariable('x', 0, 4)
    y = LpVariable('y', -1, 1)
    z = LpVariable('z', 0)
    w = LpVariable('w', 0)
    prob += (x + 4 * y + 9 * z, 'obj')
    prob += (x + y <= 5, 'c1')
    prob += (x + z >= 10, 'c2')
    prob += (-y + z == 7, 'c3')
    prob += (w >= 0, 'c4')
    filename = name + '.mps'
    prob.writeMPS(filename)
    _vars, prob2 = LpProblem.fromMPS(filename, sense=prob.sense)
    _dict1 = getSortedDict(prob)
    _dict2 = getSortedDict(prob2)
    print('\t Testing reading MPS files - noname')
    self.assertDictEqual(_dict1, _dict2)