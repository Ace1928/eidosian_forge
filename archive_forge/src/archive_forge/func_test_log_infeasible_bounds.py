import logging
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.environ import ConcreteModel, Constraint, Var, inequality
from pyomo.util.infeasible import (
def test_log_infeasible_bounds(self):
    """Test for logging of infeasible variable bounds."""
    m = self.build_model()
    m.x.setlb(2)
    m.x.setub(0)
    with LoggingIntercept(None, 'pyomo.util.infeasible') as LOG:
        log_infeasible_bounds(m)
    self.assertEqual('log_infeasible_bounds() called with a logger whose effective level is higher than logging.INFO: no output will be logged regardless of bound feasibility', LOG.getvalue().strip())
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.util', logging.INFO):
        log_infeasible_bounds(m)
    expected_output = ['VAR x: LB 2 </= 1', 'VAR x: 1 </= UB 0', 'VAR z: no assigned value.', 'VAR y4: 2 </= UB 1']
    self.assertEqual(expected_output, output.getvalue().splitlines())