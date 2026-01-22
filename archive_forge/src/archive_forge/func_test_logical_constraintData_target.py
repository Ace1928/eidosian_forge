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
def test_logical_constraintData_target(self):
    m = self.make_indexed_logical_constraint_model()
    TransformationFactory('core.logical_to_linear').apply_to(m, targets=m.cons[2])
    _constrs_contained_within(self, [(1, m.Y[2].get_associated_binary() + m.Y[3].get_associated_binary() + (1 - m.Y[1].get_associated_binary()), None)], m.logic_to_linear.transformed_constraints)
    self.assertEqual(len(m.logic_to_linear.transformed_constraints), 1)