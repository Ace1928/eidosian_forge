import pyomo.common.unittest as unittest
from pyomo.common.errors import MouseTrap, DeveloperError
from pyomo.common.log import LoggingIntercept
import logging
from pyomo.core.expr.sympy_tools import sympy_available
from pyomo.core.plugins.transform.logical_to_linear import (
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.repn import generate_standard_repn
from io import StringIO
def test_backmap_noninteger(self):
    m = _generate_boolean_model(2)
    TransformationFactory('core.logical_to_linear').apply_to(m)
    m.Y_asbinary[1].value = 0.9
    update_boolean_vars_from_binary(m, integer_tolerance=0.1)
    self.assertTrue(m.Y[1].value)
    with self.assertRaisesRegex(ValueError, 'Binary variable has non-\\{0,1\\} value'):
        update_boolean_vars_from_binary(m)