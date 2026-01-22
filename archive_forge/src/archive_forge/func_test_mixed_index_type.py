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
def test_mixed_index_type(self):
    m = ConcreteModel()
    m.I = Set(initialize=[1, '1', 3.5, 4])
    m.x = Var(m.I)
    v = list(m.component_data_objects(Var, sort=True))
    self.assertEqual(len(v), 4)
    for a, b in zip([m.x[1], m.x[3.5], m.x[4], m.x['1']], v):
        self.assertIs(a, b)