import logging
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.environ import ConcreteModel, Constraint, Var, inequality
from pyomo.util.infeasible import (
def test_log_infeasible_constraints_verbose_variables(self):
    """Test for logging of infeasible constraints."""
    m = self.build_model()
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.util.infeasible', logging.INFO):
        log_infeasible_constraints(m, log_variables=True)
    expected_output = ['CONSTR c1: 2.0 </= 1', '  - VAR x: 1', 'CONSTR c2: 1 =/= 4.0', '  - VAR x: 1', 'CONSTR c3: 1 </= 0.0', '  - VAR x: 1', 'CONSTR c5: 5.0 <?= evaluation error <?= 10.0', '  - VAR z: None', 'CONSTR c7: evaluation error =?= 6.0', '  - VAR z: None', 'CONSTR c8: 3.0 </= 1 <= 6.0', '  - VAR x: 1', 'CONSTR c9: 0.0 <= 1 </= 0.5', '  - VAR x: 1']
    self.assertEqual(expected_output, output.getvalue().splitlines())