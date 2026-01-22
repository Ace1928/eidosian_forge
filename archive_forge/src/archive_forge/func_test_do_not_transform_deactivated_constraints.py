import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet
from pyomo.contrib.preprocessing.plugins.var_aggregator import (
from pyomo.environ import (
def test_do_not_transform_deactivated_constraints(self):
    m = ConcreteModel()
    m.x = Var()
    m.y = Var()
    m.c1 = Constraint(expr=m.x == m.y)
    m.c2 = Constraint(expr=(2, m.x, 3))
    m.c3 = Constraint(expr=m.x == 0)
    m.c3.deactivate()
    TransformationFactory('contrib.aggregate_vars').apply_to(m)
    self.assertIs(m.c2.body, m._var_aggregator_info.z[1])
    self.assertIs(m.c3.body, m.x)