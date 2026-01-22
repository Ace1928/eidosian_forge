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
def test_importMPS_binary(self):
    name = self._testMethodName
    prob = LpProblem(name, const.LpMaximize)
    dummy = LpVariable('dummy')
    c1 = LpVariable('c1', 0, 1, const.LpBinary)
    c2 = LpVariable('c2', 0, 1, const.LpBinary)
    prob += dummy
    prob += c1 + c2 == 2
    prob += c1 <= 0
    filename = name + '.mps'
    prob.writeMPS(filename)
    _vars, prob2 = LpProblem.fromMPS(filename, sense=prob.sense, dropConsNames=True)
    _dict1 = getSortedDict(prob, keyCons='constant')
    _dict2 = getSortedDict(prob2, keyCons='constant')
    print('\t Testing reading MPS files - binary variable, no constraint names')
    self.assertDictEqual(_dict1, _dict2)