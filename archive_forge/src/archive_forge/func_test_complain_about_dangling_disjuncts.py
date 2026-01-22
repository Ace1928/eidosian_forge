from pyomo.common.errors import InfeasibleConstraintException
import pyomo.common.unittest as unittest
from pyomo.environ import Block, ConcreteModel, Constraint, TransformationFactory
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import GDP_Error
def test_complain_about_dangling_disjuncts(self):
    m = ConcreteModel()
    m.d = Disjunct([1, 2, 3, 4])
    m.disj1 = Disjunction(expr=[m.d[1], m.d[2], m.d[3]])
    m.d[1].indicator_var = True
    with self.assertRaisesRegex(GDP_Error, 'Found active Disjuncts on the model that were not included in any Disjunctions:\\nd\\[4\\]\\nPlease deactivate them or include them in a Disjunction.'):
        reverse = TransformationFactory('gdp.transform_current_disjunctive_state').apply_to(m)