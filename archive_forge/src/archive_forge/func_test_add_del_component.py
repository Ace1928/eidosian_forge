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
def test_add_del_component(self):
    m = Block()
    self.assertFalse(m.contains_component(Var))
    m.x = x = Var()
    self.assertTrue(m.contains_component(Var))
    self.assertTrue('x' in m.__dict__)
    self.assertIs(m.component('x'), x)
    del m.x
    self.assertFalse(m.contains_component(Var))
    self.assertFalse('x' in m.__dict__)
    self.assertIs(m.component('x'), None)