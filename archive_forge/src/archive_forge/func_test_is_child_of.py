import pyomo.common.unittest as unittest
from pyomo.core import ConcreteModel, Var, Expression, Block, RangeSet, Any
import pyomo.core.expr as EXPR
from pyomo.core.base.expression import _ExpressionData
from pyomo.gdp.util import (
from pyomo.gdp import Disjunct, Disjunction
def test_is_child_of(self):
    m = ConcreteModel()
    m.b = Block()
    m.b.b_indexed = Block([1, 2])
    m.b_parallel = Block()
    knownBlocks = {}
    self.assertFalse(is_child_of(parent=m.b, child=m.b_parallel, knownBlocks=knownBlocks))
    self.assertEqual(len(knownBlocks), 2)
    self.assertFalse(knownBlocks.get(m))
    self.assertFalse(knownBlocks.get(m.b_parallel))
    self.assertTrue(is_child_of(parent=m.b, child=m.b.b_indexed[1], knownBlocks=knownBlocks))
    self.assertEqual(len(knownBlocks), 4)
    self.assertFalse(knownBlocks.get(m))
    self.assertFalse(knownBlocks.get(m.b_parallel))
    self.assertTrue(knownBlocks.get(m.b.b_indexed[1]))
    self.assertTrue(knownBlocks.get(m.b.b_indexed))