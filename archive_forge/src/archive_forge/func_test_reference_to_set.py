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
def test_reference_to_set(self):
    m = ConcreteModel()
    m.I = Set(initialize=[1, 3, 5])
    m.r = Reference(m.I)
    self.assertEqual(len(m.r), 1)
    self.assertEqual(list(m.r.keys()), [None])
    self.assertEqual(list(m.r.values()), [m.I])
    self.assertIs(m.r[None], m.I)
    m = ConcreteModel()
    m.I = Set(initialize=[1, 3, None, 5])
    m.r = Reference(m.I)
    self.assertEqual(len(m.r), 1)
    self.assertEqual(list(m.r.keys()), [None])
    self.assertEqual(list(m.r.values()), [m.I])
    self.assertIs(m.r[None], m.I)