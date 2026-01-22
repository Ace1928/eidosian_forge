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
def test_replace_attribute_with_component(self):
    OUTPUT = StringIO()
    with LoggingIntercept(OUTPUT, 'pyomo.core'):
        self.block.x = 5
        self.block.x = Var()
    self.assertIn('Reassigning the non-component attribute', OUTPUT.getvalue())