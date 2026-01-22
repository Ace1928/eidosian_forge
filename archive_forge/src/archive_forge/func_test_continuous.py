import pyomo.common.unittest as unittest
from io import StringIO
from pyomo.environ import (
from pyomo.network import Port, Arc
def test_continuous(self):
    m = ConcreteModel()
    m.x = Var(domain=Reals)
    m.y = Var(domain=Integers)
    m.p = Port()
    self.assertTrue(m.p.is_continuous())
    m.p.add(m.x)
    self.assertTrue(m.p.is_continuous())
    m.p.add(-m.x, 'foo')
    self.assertTrue(m.p.is_continuous())
    m.p.add(m.y)
    self.assertFalse(m.p.is_continuous())
    m.p.remove('y')
    self.assertTrue(m.p.is_continuous())
    m.p.add(-m.y, 'bar')
    self.assertFalse(m.p.is_continuous())