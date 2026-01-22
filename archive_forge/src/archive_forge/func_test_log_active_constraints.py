import logging
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.environ import ConcreteModel, Constraint, Var, inequality
from pyomo.util.infeasible import (
def test_log_active_constraints(self):
    """Test for logging of active constraints."""
    m = self.build_model()
    depr = StringIO()
    output = StringIO()
    with LoggingIntercept(depr, 'pyomo.util', logging.WARNING):
        log_active_constraints(m)
    self.assertIn('log_active_constraints is deprecated.', depr.getvalue())
    with LoggingIntercept(output, 'pyomo.util', logging.INFO):
        log_active_constraints(m)
    expected_output = ['c1 active', 'c2 active', 'c3 active', 'c4 active', 'c5 active', 'c6 active', 'c7 active', 'c8 active', 'c9 active', 'c11 active']
    self.assertEqual(expected_output, output.getvalue()[len(depr.getvalue()):].splitlines())