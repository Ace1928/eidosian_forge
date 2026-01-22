from pyomo.common.errors import InfeasibleConstraintException
import pyomo.common.unittest as unittest
from pyomo.environ import Block, ConcreteModel, Constraint, TransformationFactory
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import GDP_Error
def test_xor_sums_to_0(self):
    m = self.make_two_term_disjunction()
    m.d1.indicator_var.set_value(False)
    m.d2.indicator_var.set_value(False)
    with self.assertRaisesRegex(InfeasibleConstraintException, "Logical constraint for Disjunction 'disjunction1' is violated: All the Disjunct indicator_vars are 'False.'"):
        TransformationFactory('gdp.transform_current_disjunctive_state').apply_to(m)