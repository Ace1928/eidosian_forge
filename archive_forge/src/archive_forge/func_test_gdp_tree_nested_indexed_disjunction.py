import pyomo.common.unittest as unittest
from pyomo.core import ConcreteModel, Var, Expression, Block, RangeSet, Any
import pyomo.core.expr as EXPR
from pyomo.core.base.expression import _ExpressionData
from pyomo.gdp.util import (
from pyomo.gdp import Disjunct, Disjunction
def test_gdp_tree_nested_indexed_disjunction(self):
    m = ConcreteModel()
    m.I = RangeSet(1, 4)
    m.x = Var(m.I, bounds=(-2, 6))
    m.disj1 = Disjunct()
    self.add_indexed_disjunction(m.disj1, m)
    m.disj2 = Disjunct()
    m.another_disjunction = Disjunction(expr=[m.disj1, m.disj2])
    targets = (m.disj1.indexed,)
    knownBlocks = {}
    tree = get_gdp_tree(targets, m, knownBlocks)
    vertices = tree.vertices
    self.assertEqual(len(vertices), 6)
    in_degrees = {m.disj1.indexed[0]: 0, m.disj1.indexed[1]: 0, m.disj1.indexed[0].disjuncts[0]: 1, m.disj1.indexed[0].disjuncts[1]: 1, m.disj1.indexed[1].disjuncts[0]: 1, m.disj1.indexed[1].disjuncts[1]: 1}
    for key, val in in_degrees.items():
        self.assertEqual(tree.in_degree(key), val)
    topo_sort = [m.disj1.indexed[0], m.disj1.indexed[0].disjuncts[1], m.disj1.indexed[0].disjuncts[0], m.disj1.indexed[1], m.disj1.indexed[1].disjuncts[1], m.disj1.indexed[1].disjuncts[0]]
    sort = tree.topological_sort()
    for i, node in enumerate(sort):
        self.assertIs(node, topo_sort[i])
    targets = (m,)
    tree = get_gdp_tree(targets, m, knownBlocks)
    vertices = tree.vertices
    self.assertEqual(len(vertices), 9)
    in_degrees[m.disj1.indexed[0]] = 1
    in_degrees[m.disj1.indexed[1]] = 1
    in_degrees[m.disj1] = 1
    in_degrees[m.disj2] = 1
    in_degrees[m.another_disjunction] = 0
    for key, val in in_degrees.items():
        self.assertEqual(tree.in_degree(key), val)
    topo_sort = [m.another_disjunction, m.disj2, m.disj1, m.disj1.indexed[1], m.disj1.indexed[1].disjuncts[1], m.disj1.indexed[1].disjuncts[0], m.disj1.indexed[0], m.disj1.indexed[0].disjuncts[1], m.disj1.indexed[0].disjuncts[0]]
    sort = tree.topological_sort()
    for i, node in enumerate(sort):
        self.assertIs(node, topo_sort[i])