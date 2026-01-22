import logging
from itertools import product
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.contrib.multistart.high_conf_stop import should_stop
from pyomo.contrib.multistart.reinit import strategies
from pyomo.environ import (
def test_model_infeasible(self):
    m = ConcreteModel()
    m.x = Var(bounds=(0, 1))
    m.c = Constraint(expr=m.x >= 2)
    m.o = Objective(expr=m.x)
    SolverFactory('multistart').solve(m, iterations=2)
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.contrib.multistart', logging.WARNING):
        SolverFactory('multistart').solve(m, iterations=-1, HCS_max_iterations=3)
        self.assertIn('High confidence stopping rule was unable to complete after 3 iterations.', output.getvalue().strip())