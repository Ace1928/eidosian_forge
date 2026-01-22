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
def test_clear_attribute(self):
    """Coverage of the _clear_attribute method"""
    obj = Set()
    self.block.A = obj
    self.assertEqual(self.block.A.local_name, 'A')
    self.assertEqual(obj.local_name, 'A')
    self.assertIs(obj, self.block.A)
    obj = Var()
    self.block.A = obj
    self.assertEqual(self.block.A.local_name, 'A')
    self.assertEqual(obj.local_name, 'A')
    self.assertIs(obj, self.block.A)
    obj = Param()
    self.block.A = obj
    self.assertEqual(self.block.A.local_name, 'A')
    self.assertEqual(obj.local_name, 'A')
    self.assertIs(obj, self.block.A)
    obj = Objective()
    self.block.A = obj
    self.assertEqual(self.block.A.local_name, 'A')
    self.assertEqual(obj.local_name, 'A')
    self.assertIs(obj, self.block.A)
    obj = Constraint()
    self.block.A = obj
    self.assertEqual(self.block.A.local_name, 'A')
    self.assertEqual(obj.local_name, 'A')
    self.assertIs(obj, self.block.A)
    obj = Set()
    self.block.A = obj
    self.assertEqual(self.block.A.local_name, 'A')
    self.assertEqual(obj.local_name, 'A')
    self.assertIs(obj, self.block.A)