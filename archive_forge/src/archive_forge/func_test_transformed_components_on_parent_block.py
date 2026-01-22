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
def test_transformed_components_on_parent_block(self):
    m = ConcreteModel()
    m.b = Block()
    m.b.s = RangeSet(3)
    m.b.Y = BooleanVar(m.b.s)
    m.b.p = LogicalConstraint(expr=m.b.Y[1].implies(lor(m.b.Y[2], m.b.Y[3])))
    TransformationFactory('core.logical_to_linear').apply_to(m)
    boolean_var = m.b.component('Y_asbinary')
    self.assertIsInstance(boolean_var, Var)
    notAVar = m.component('Y_asbinary')
    self.assertIsNone(notAVar)
    transBlock = m.b.component('logic_to_linear')
    self.assertIsInstance(transBlock, Block)
    notAThing = m.component('logic_to_linear')
    self.assertIsNone(notAThing)
    _constrs_contained_within(self, [(1, m.b.Y[2].get_associated_binary() + m.b.Y[3].get_associated_binary() + (1 - m.b.Y[1].get_associated_binary()), None)], m.b.logic_to_linear.transformed_constraints)