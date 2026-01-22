import pyomo.common.unittest as unittest
from pyomo.core.expr.symbol_map import SymbolMap
from pyomo.core.kernel.variable import variable
from pyomo.environ import ConcreteModel, Var
def test_existing_alias(self):
    s = SymbolMap()
    v1 = variable()
    s.alias(v1, 'v')
    self.assertIs(s.aliases['v'], v1)
    v2 = variable()
    with self.assertRaises(RuntimeError):
        s.alias(v2, 'v')
    s.alias(v1, 'A')
    self.assertIs(s.aliases['v'], v1)
    self.assertIs(s.aliases['A'], v1)
    s.alias(v1, 'A')
    self.assertIs(s.aliases['v'], v1)
    self.assertIs(s.aliases['A'], v1)