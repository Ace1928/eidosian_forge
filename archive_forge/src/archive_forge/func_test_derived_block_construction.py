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
def test_derived_block_construction(self):

    class ConcreteBlock(Block):
        pass

    class ScalarConcreteBlock(_BlockData, ConcreteBlock):

        def __init__(self, *args, **kwds):
            _BlockData.__init__(self, component=self)
            ConcreteBlock.__init__(self, *args, **kwds)
    _buf = []

    def _rule(b):
        _buf.append(1)
    m = ConcreteModel()
    m.b = ScalarConcreteBlock(rule=_rule)
    self.assertEqual(_buf, [1])