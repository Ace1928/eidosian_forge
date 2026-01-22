from pyomo.common.errors import MouseTrap
import pyomo.common.unittest as unittest
from pyomo.contrib.cp.transform.logical_to_disjunctive_program import (
from pyomo.contrib.cp.transform.logical_to_disjunctive_walker import (
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.core.plugins.transform.logical_to_linear import (
from pyomo.gdp import Disjunct
from pyomo.environ import (
def test_integer_var_in_at_least(self):
    m = self.make_model()
    m.x = Var(bounds=(0, 10), domain=Integers)
    e = atleast(m.x, m.a, m.b, m.c)
    visitor = LogicalToDisjunctiveVisitor()
    m.cons = visitor.constraints
    m.z = visitor.z_vars
    with self.assertRaisesRegex(MouseTrap, "The first argument 'x' to 'atleast\\(x: \\[a, b, c\\]\\)' is potentially variable. This may be a mathematically coherent expression; However it is not yet supported to convert it to a disjunctive program.", normalize_whitespace=True):
        visitor.walk_expression(e)