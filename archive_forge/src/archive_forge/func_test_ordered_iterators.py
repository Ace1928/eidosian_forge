import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from io import StringIO
from pyomo.environ import (
from pyomo.common.collections import ComponentSet
from pyomo.common.log import LoggingIntercept
from pyomo.core.base.var import IndexedVar
from pyomo.core.base.set import (
from pyomo.core.base.indexed_component import UnindexedComponent_set, IndexedComponent
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.reference import (
def test_ordered_iterators(self):
    m = ConcreteModel()
    m.I = Set(initialize=[3, 2])
    m.b = Block([1, 0])
    m.b[1].x = Var(m.I)
    m.b[0].x = Var(m.I)
    m.y = Reference(m.b[:].x[:])
    self.assertEqual(list(m.y.index_set().subsets()), [m.b.index_set(), m.I])
    self.assertEqual(list(m.y), [(1, 3), (1, 2), (0, 3), (0, 2)])
    self.assertEqual(list(m.y.keys()), [(1, 3), (1, 2), (0, 3), (0, 2)])
    self.assertEqual(list(m.y.values()), [m.b[1].x[3], m.b[1].x[2], m.b[0].x[3], m.b[0].x[2]])
    self.assertEqual(list(m.y.items()), [((1, 3), m.b[1].x[3]), ((1, 2), m.b[1].x[2]), ((0, 3), m.b[0].x[3]), ((0, 2), m.b[0].x[2])])
    self.assertEqual(list(m.y.keys(True)), [(0, 2), (0, 3), (1, 2), (1, 3)])
    self.assertEqual(list(m.y.values(True)), [m.b[0].x[2], m.b[0].x[3], m.b[1].x[2], m.b[1].x[3]])
    self.assertEqual(list(m.y.items(True)), [((0, 2), m.b[0].x[2]), ((0, 3), m.b[0].x[3]), ((1, 2), m.b[1].x[2]), ((1, 3), m.b[1].x[3])])
    m = ConcreteModel()
    m.b = Block([1, 0])
    m.b[1].x = Var([3, 2])
    m.b[0].x = Var([5, 4])
    m.y = Reference(m.b[:].x[:])
    self.assertIs(type(m.y.index_set()), FiniteSetOf)
    self.assertEqual(list(m.y), [(1, 3), (1, 2), (0, 5), (0, 4)])
    self.assertEqual(list(m.y.keys()), [(1, 3), (1, 2), (0, 5), (0, 4)])
    self.assertEqual(list(m.y.values()), [m.b[1].x[3], m.b[1].x[2], m.b[0].x[5], m.b[0].x[4]])
    self.assertEqual(list(m.y.items()), [((1, 3), m.b[1].x[3]), ((1, 2), m.b[1].x[2]), ((0, 5), m.b[0].x[5]), ((0, 4), m.b[0].x[4])])
    self.assertEqual(list(m.y.keys(True)), [(0, 4), (0, 5), (1, 2), (1, 3)])
    self.assertEqual(list(m.y.values(True)), [m.b[0].x[4], m.b[0].x[5], m.b[1].x[2], m.b[1].x[3]])
    self.assertEqual(list(m.y.items(True)), [((0, 4), m.b[0].x[4]), ((0, 5), m.b[0].x[5]), ((1, 2), m.b[1].x[2]), ((1, 3), m.b[1].x[3])])
    m = ConcreteModel()
    m.b = Block([1, 0])
    m.b[1].x = Var([3, 2])
    m.b[0].x = Var([5, 4])
    m.y = Reference({(1, 3): m.b[1].x[3], (0, 5): m.b[0].x[5], (1, 2): m.b[1].x[2], (0, 4): m.b[0].x[4]})
    self.assertIs(type(m.y.index_set()), FiniteSetOf)
    self.assertEqual(list(m.y), [(1, 3), (0, 5), (1, 2), (0, 4)])
    self.assertEqual(list(m.y.keys()), [(1, 3), (0, 5), (1, 2), (0, 4)])
    self.assertEqual(list(m.y.values()), [m.b[1].x[3], m.b[0].x[5], m.b[1].x[2], m.b[0].x[4]])
    self.assertEqual(list(m.y.items()), [((1, 3), m.b[1].x[3]), ((0, 5), m.b[0].x[5]), ((1, 2), m.b[1].x[2]), ((0, 4), m.b[0].x[4])])
    self.assertEqual(list(m.y.keys(True)), [(0, 4), (0, 5), (1, 2), (1, 3)])
    self.assertEqual(list(m.y.values(True)), [m.b[0].x[4], m.b[0].x[5], m.b[1].x[2], m.b[1].x[3]])
    self.assertEqual(list(m.y.items(True)), [((0, 4), m.b[0].x[4]), ((0, 5), m.b[0].x[5]), ((1, 2), m.b[1].x[2]), ((1, 3), m.b[1].x[3])])