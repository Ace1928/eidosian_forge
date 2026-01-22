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
def test_link_with_gdp_indicators(self):
    m = _generate_boolean_model(4)
    m.d1 = Disjunct()
    m.d2 = Disjunct()
    m.x = Var()
    m.dd = Disjunct([1, 2])
    m.d1.c = Constraint(expr=m.x >= 2)
    m.d2.c = Constraint(expr=m.x <= 10)
    m.dd[1].c = Constraint(expr=m.x >= 5)
    m.dd[2].c = Constraint(expr=m.x <= 6)
    m.Y[1].associate_binary_var(m.d1.binary_indicator_var)
    m.Y[2].associate_binary_var(m.d2.binary_indicator_var)
    m.Y[3].associate_binary_var(m.dd[1].binary_indicator_var)
    m.Y[4].associate_binary_var(m.dd[2].binary_indicator_var)
    m.p = LogicalConstraint(expr=m.Y[1].implies(lor(m.Y[3], m.Y[4])))
    m.p2 = LogicalConstraint(expr=atmost(2, *m.Y[:]))
    TransformationFactory('core.logical_to_linear').apply_to(m)
    _constrs_contained_within(self, [(1, m.dd[1].binary_indicator_var + m.dd[2].binary_indicator_var + 1 - m.d1.binary_indicator_var, None), (None, m.d1.binary_indicator_var + m.d2.binary_indicator_var + m.dd[1].binary_indicator_var + m.dd[2].binary_indicator_var, 2)], m.logic_to_linear.transformed_constraints)