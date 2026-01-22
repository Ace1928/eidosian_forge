from io import StringIO
import os
import sys
import types
import json
from copy import deepcopy
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.core.base.block import (
import pyomo.core.expr as EXPR
from pyomo.opt import check_available_solvers
from pyomo.gdp import Disjunct
def test_abstract_transfer_construction(self):
    m = AbstractModel()
    m.I = Set()

    def b_rule(_b, i):
        b = Block()
        b.p = Param(default=i)
        b.J = Set(initialize=range(i))
        return b
    m.b = Block(m.I, rule=b_rule)
    i = m.create_instance({None: {'I': {None: [1, 2, 3, 4]}, 'b': {1: {'p': {None: 10}, 'J': {None: [7, 8]}}, 2: {'p': {None: 12}}, 3: {'J': {None: [9]}}}}})
    self.assertEqual(list(i.I), [1, 2, 3, 4])
    self.assertEqual(len(i.b), 4)
    self.assertEqual(list(i.b[1].J), [7, 8])
    self.assertEqual(list(i.b[2].J), [0, 1])
    self.assertEqual(list(i.b[3].J), [9])
    self.assertEqual(list(i.b[4].J), [0, 1, 2, 3])
    self.assertEqual(value(i.b[1].p), 10)
    self.assertEqual(value(i.b[2].p), 12)
    self.assertEqual(value(i.b[3].p), 3)
    self.assertEqual(value(i.b[4].p), 4)