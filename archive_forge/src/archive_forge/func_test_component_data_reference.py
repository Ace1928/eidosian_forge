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
def test_component_data_reference(self):
    m = ConcreteModel()
    m.y = Var([1, 2])
    m.r = Reference(m.y[2])
    self.assertIs(m.r.ctype, Var)
    self.assertIsNot(m.r.index_set(), m.y.index_set())
    self.assertIs(m.r.index_set(), UnindexedComponent_ReferenceSet)
    self.assertEqual(len(m.r), 1)
    self.assertTrue(m.r.is_reference())
    self.assertTrue(m.r.is_indexed())
    self.assertIn(None, m.r)
    self.assertNotIn(1, m.r)
    self.assertIs(m.r[None], m.y[2])
    with self.assertRaises(KeyError):
        m.r[2]