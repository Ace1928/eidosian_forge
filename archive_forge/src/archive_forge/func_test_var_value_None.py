import logging
from itertools import product
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.contrib.multistart.high_conf_stop import should_stop
from pyomo.contrib.multistart.reinit import strategies
from pyomo.environ import (
def test_var_value_None(self):
    m = ConcreteModel()
    m.x = Var(bounds=(0, 1))
    m.obj = Objective(expr=m.x)
    SolverFactory('multistart').solve(m)