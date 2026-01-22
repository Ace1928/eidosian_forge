import pyomo.common.unittest as unittest
from pyomo.contrib.preprocessing.plugins.induced_linearity import (
from pyomo.common.collections import ComponentSet, Bunch
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.repn import generate_standard_repn
@unittest.skipIf(not glpk_available, 'GLPK not available')
def test_bilinear_in_disjuncts(self):
    m = ConcreteModel()
    m.x = Var([0], bounds=(-3, 8))
    m.y = Var(RangeSet(4), domain=Binary)
    m.z = Var(domain=Integers, bounds=(-1, 2))
    m.constr = Constraint(expr=m.x[0] == m.y[1] + 2 * m.y[2] + m.y[3] + 2 * m.y[4] + m.z)
    m.logical = ConstraintList()
    m.logical.add(expr=m.y[1] + m.y[2] == 1)
    m.logical.add(expr=m.y[3] + m.y[4] == 1)
    m.logical.add(expr=m.y[2] + m.y[4] <= 1)
    m.v = Var([1, 2])
    m.v[1].setlb(-2)
    m.v[1].setub(7)
    m.v[2].setlb(-4)
    m.v[2].setub(5)
    m.bilinear = Constraint(expr=(m.x[0] - 3) * (m.v[1] + 2) - (m.v[2] + 4) * m.v[1] + exp(m.v[1] ** 2) * m.x[0] <= m.v[2])
    m.disjctn = Disjunction(expr=[[m.x[0] * m.v[1] <= 4], [m.x[0] * m.v[2] >= 6]])
    TransformationFactory('contrib.induced_linearity').apply_to(m)
    self.assertEqual(m.disjctn.disjuncts[0].constraint[1].body.polynomial_degree(), 1)
    self.assertEqual(m.disjctn.disjuncts[1].constraint[1].body.polynomial_degree(), 1)