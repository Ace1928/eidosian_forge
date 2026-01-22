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
def test_blockData_target(self):
    m = ConcreteModel()
    m.b = Block([1, 2])
    m.b[1].transfer_attributes_from(self.make_indexed_logical_constraint_model())
    TransformationFactory('core.logical_to_linear').apply_to(m, targets=m.b[1])
    _constrs_contained_within(self, [(2, m.b[1].Y[1].get_associated_binary() + m.b[1].Y[2].get_associated_binary() + m.b[1].Y[3].get_associated_binary(), 2)], m.b[1].logic_to_linear.transformed_constraints)
    _constrs_contained_within(self, [(1, m.b[1].Y[2].get_associated_binary() + m.b[1].Y[3].get_associated_binary() + (1 - m.b[1].Y[1].get_associated_binary()), None)], m.b[1].logic_to_linear.transformed_constraints)