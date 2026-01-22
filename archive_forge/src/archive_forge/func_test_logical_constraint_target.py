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
def test_logical_constraint_target(self):
    m = _generate_boolean_model(3)
    TransformationFactory('core.logical_to_linear').apply_to(m, targets=m.constraint)
    _constrs_contained_within(self, [(2, m.Y[1].get_associated_binary() + m.Y[2].get_associated_binary() + m.Y[3].get_associated_binary(), 2)], m.logic_to_linear.transformed_constraints)