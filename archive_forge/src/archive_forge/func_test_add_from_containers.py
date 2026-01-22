import pyomo.common.unittest as unittest
from io import StringIO
from pyomo.environ import (
from pyomo.network import Port, Arc
def test_add_from_containers(self):
    m = ConcreteModel()
    m.x = Var()
    m.y = Var()
    m.p1 = Port(initialize=[m.x, m.y])
    m.p2 = Port(initialize=[(m.x, Port.Equality), (m.y, Port.Extensive)])
    m.p3 = Port(initialize=dict(this=m.x, that=m.y))
    m.p4 = Port(initialize=dict(this=(m.x, Port.Equality), that=(m.y, Port.Extensive)))
    self.assertIs(m.p1.x, m.x)
    self.assertIs(m.p1.y, m.y)
    self.assertIs(m.p2.x, m.x)
    self.assertTrue(m.p2.is_equality('x'))
    self.assertIs(m.p2.y, m.y)
    self.assertTrue(m.p2.is_extensive('y'))
    self.assertIs(m.p3.this, m.x)
    self.assertIs(m.p3.that, m.y)
    self.assertIs(m.p4.this, m.x)
    self.assertTrue(m.p4.is_equality('this'))
    self.assertIs(m.p4.that, m.y)
    self.assertTrue(m.p4.is_extensive('that'))