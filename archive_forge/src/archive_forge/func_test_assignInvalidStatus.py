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
def test_assignInvalidStatus(self):
    print('\t Testing invalid status')
    t = LpProblem('test')
    Invalid = -100
    self.assertRaises(const.PulpError, lambda: t.assignStatus(Invalid))
    self.assertRaises(const.PulpError, lambda: t.assignStatus(0, Invalid))