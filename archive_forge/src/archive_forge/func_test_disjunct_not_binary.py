import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.opt import check_available_solvers
def test_disjunct_not_binary(self):
    m = ConcreteModel()
    m.d1 = Disjunct()
    m.d2 = Disjunct()
    m.d = Disjunction(expr=[m.d1, m.d2])
    m.d1.binary_indicator_var.domain = NonNegativeReals
    m.d2.binary_indicator_var.domain = NonNegativeReals
    m.d1.binary_indicator_var.set_value(0.5)
    m.d2.binary_indicator_var.set_value(0.5)
    with self.assertRaisesRegex(GDP_Error, "The value of the indicator_var of Disjunct 'd1' is None. All indicator_vars must have values before calling 'fix_disjuncts'."):
        TransformationFactory('gdp.fix_disjuncts').apply_to(m)