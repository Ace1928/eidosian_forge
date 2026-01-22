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
def test_reference_to_sparse(self):
    m = ConcreteModel()
    m.I = RangeSet(3)
    m.x = Var(m.I, m.I, dense=False)
    m.xx = Reference(m.x[...], ctype=Var)
    self.assertEqual(len(m.x), 0)
    self.assertNotIn((1, 1), m.x)
    self.assertNotIn((1, 1), m.xx)
    self.assertIn((1, 1), m.x.index_set())
    self.assertIn((1, 1), m.xx.index_set())
    self.assertEqual(len(m.x), 0)
    m.xx[1, 2]
    self.assertEqual(len(m.x), 1)
    self.assertIs(m.xx[1, 2], m.x[1, 2])
    self.assertEqual(len(m.x), 1)
    m.xx[1, 3] = 5
    self.assertEqual(len(m.x), 2)
    self.assertIs(m.xx[1, 3], m.x[1, 3])
    self.assertEqual(len(m.x), 2)
    self.assertEqual(value(m.x[1, 3]), 5)
    m.xx.add((1, 1))
    self.assertEqual(len(m.x), 3)
    self.assertIs(m.xx[1, 1], m.x[1, 1])
    self.assertEqual(len(m.x), 3)