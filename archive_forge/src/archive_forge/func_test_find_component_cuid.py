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
def test_find_component_cuid(self):
    b = Block(concrete=True)
    b.v1 = Var()
    b.v2 = Var([1, 2])
    cuid1 = ComponentUID('v1')
    cuid2 = ComponentUID('v2[2]')
    self.assertIs(b.find_component(cuid1), b.v1)
    self.assertIs(b.find_component(cuid2), b.v2[2])