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
def test_false_constraint(self):
    prob = LpProblem(self._testMethodName, const.LpMinimize)

    def add_const(prob):
        prob += 0 - 3 == 0
    self.assertRaises(TypeError, add_const, prob=prob)