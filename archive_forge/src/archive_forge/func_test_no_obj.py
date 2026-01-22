import logging
from itertools import product
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.contrib.multistart.high_conf_stop import should_stop
from pyomo.contrib.multistart.reinit import strategies
from pyomo.environ import (
def test_no_obj(self):
    m = ConcreteModel()
    m.x = Var()
    with self.assertRaisesRegex(RuntimeError, 'no active objective'):
        SolverFactory('multistart').solve(m)