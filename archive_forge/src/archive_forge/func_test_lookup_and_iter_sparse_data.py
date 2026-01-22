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
def test_lookup_and_iter_sparse_data(self):
    m = ConcreteModel()
    m.I = RangeSet(3)
    m.x = Var(m.I, m.I, dense=False)
    rd = _ReferenceDict(m.x[...])
    rs = _ReferenceSet(m.x[...])
    self.assertEqual(len(rd), 0)
    self.assertEqual(len(rd), 0)
    self.assertEqual(len(rs), 9)
    self.assertEqual(len(rd), 0)
    self.assertIn((1, 1), rs)
    self.assertEqual(len(rd), 0)
    self.assertEqual(len(rs), 9)