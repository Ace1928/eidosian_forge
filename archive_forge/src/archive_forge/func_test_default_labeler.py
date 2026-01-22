import pyomo.common.unittest as unittest
from pyomo.core.expr.symbol_map import SymbolMap
from pyomo.core.kernel.variable import variable
from pyomo.environ import ConcreteModel, Var
def test_default_labeler(self):
    s = SymbolMap(lambda x: '_' + str(x))
    v = variable()
    self.assertEqual('_' + str(v), s.getSymbol(v))
    s = SymbolMap(lambda x: '_' + str(x))
    m = ConcreteModel()
    m.x = Var()
    self.assertEqual('_x', s.createSymbol(m.x))
    s = SymbolMap(lambda x: '_' + str(x))
    m.y = Var([1, 2, 3])
    s.createSymbols(m.y.values())
    self.assertEqual(s.bySymbol, {'_y[1]': m.y[1], '_y[2]': m.y[2], '_y[3]': m.y[3]})
    self.assertEqual(s.byObject, {id(m.y[1]): '_y[1]', id(m.y[2]): '_y[2]', id(m.y[3]): '_y[3]'})