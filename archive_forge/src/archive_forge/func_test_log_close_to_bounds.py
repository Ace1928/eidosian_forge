import logging
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.environ import ConcreteModel, Constraint, Var, inequality
from pyomo.util.infeasible import (
def test_log_close_to_bounds(self):
    """Test logging of variables and constraints near bounds."""
    m = self.build_model()
    with LoggingIntercept(None, 'pyomo.util.infeasible') as LOG:
        log_close_to_bounds(m)
    self.assertEqual('log_close_to_bounds() called with a logger whose effective level is higher than logging.INFO: no output will be logged regardless of bound status', LOG.getvalue().strip())
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.util.infeasible', logging.INFO):
        log_close_to_bounds(m)
    expected_output = ['y near UB of 2', 'yy near LB of 0', 'c4 near LB of 1.9999999', 'c11 near UB of 1.9999999']
    self.assertEqual(expected_output, output.getvalue().splitlines())