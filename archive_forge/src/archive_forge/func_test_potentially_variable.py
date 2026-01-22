import pyomo.common.unittest as unittest
from io import StringIO
from pyomo.environ import (
from pyomo.network import Port, Arc
def test_potentially_variable(self):
    m = ConcreteModel()
    m.x = Var()
    m.p = Port()
    self.assertTrue(m.p.is_potentially_variable())
    m.p.add(-m.x)
    self.assertTrue(m.p.is_potentially_variable())