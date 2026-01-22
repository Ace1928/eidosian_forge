import pyomo.common.unittest as unittest
from pyomo.contrib.preprocessing.plugins.induced_linearity import (
from pyomo.common.collections import ComponentSet, Bunch
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.repn import generate_standard_repn
@unittest.skipIf(not glpk_available, 'GLPK not available')
def test_induced_linearity_case2(self):
    m = ConcreteModel()
    m.x = Var([0], bounds=(-3, 8))
    m.y = Var(RangeSet(4), domain=Binary)
    m.z = Var(domain=Integers, bounds=(-1, 2))
    m.constr = Constraint(expr=m.x[0] == m.y[1] + 2 * m.y[2] + m.y[3] + 2 * m.y[4] + m.z)
    m.logical = ConstraintList()
    m.logical.add(expr=m.y[1] + m.y[2] == 1)
    m.logical.add(expr=m.y[3] + m.y[4] == 1)
    m.logical.add(expr=m.y[2] + m.y[4] <= 1)
    m.b = Var(bounds=(-2, 7))
    m.c = Var()
    m.bilinear = Constraint(expr=(m.x[0] - 3) * (m.b + 2) - (m.c + 4) * m.b + exp(m.b ** 2) * m.x[0] <= m.c)
    TransformationFactory('contrib.induced_linearity').apply_to(m)
    xfrmed_blk = m._induced_linearity_info.x0_b_bilinear
    self.assertSetEqual(set(xfrmed_blk.valid_values), set([1, 2, 3, 4, 5]))
    select_one_repn = generate_standard_repn(xfrmed_blk.select_one_value.body)
    self.assertEqual(ComponentSet(select_one_repn.linear_vars), ComponentSet((xfrmed_blk.x_active[i] for i in xfrmed_blk.valid_values)))