import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.base.symbol_map import symbol_map_from_instance
def test_creates(self):
    smap = SymbolMap()
    labeler = TextLabeler()
    smap.createSymbols(self.instance.component_data_objects(Var), labeler)
    self.assertEqual(set(smap.bySymbol.keys()), set(['x', 'y(1)', 'y(2)', 'y(3)', 'b_x', 'b_y(1)', 'b_y(2)', 'b_y(3)']))