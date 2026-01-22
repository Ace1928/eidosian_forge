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
def test_makeDict_behavior(self):
    """
            Test if makeDict is returning the expected value.
            """
    headers = [['A', 'B'], ['C', 'D']]
    values = [[1, 2], [3, 4]]
    target = {'A': {'C': 1, 'D': 2}, 'B': {'C': 3, 'D': 4}}
    dict_with_default = makeDict(headers, values, default=0)
    dict_without_default = makeDict(headers, values)
    print('\t Testing makeDict general behavior')
    self.assertEqual(dict_with_default, target)
    self.assertEqual(dict_without_default, target)