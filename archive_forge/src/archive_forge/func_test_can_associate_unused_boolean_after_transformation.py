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
def test_can_associate_unused_boolean_after_transformation(self):
    m = ConcreteModel()
    m.Y = BooleanVar()
    TransformationFactory('core.logical_to_linear').apply_to(m)
    m.y = Var(domain=Binary)
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.core.base', logging.WARNING):
        m.Y.associate_binary_var(m.y)
        y = m.Y.get_associated_binary()
    self.assertIs(y, m.y)
    self.assertEqual(output.getvalue(), '')