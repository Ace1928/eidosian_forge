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
def test_insert_var(self):
    m = ConcreteModel()
    m.T = Set(initialize=[1, 5])
    m.x = Var(m.T, initialize=lambda m, i: i)

    @m.Block(m.T)
    def b(b, i):
        b.y = Var(initialize=lambda b: 10 * b.index())
    ref_x = Reference(m.x[:])
    ref_y = Reference(m.b[:].y)
    self.assertEqual(len(m.x), 2)
    self.assertEqual(len(ref_x), 2)
    self.assertEqual(len(m.b), 2)
    self.assertEqual(len(ref_y), 2)
    self.assertEqual(value(ref_x[1]), 1)
    self.assertEqual(value(ref_x[5]), 5)
    self.assertEqual(value(ref_y[1]), 10)
    self.assertEqual(value(ref_y[5]), 50)
    m.T.add(2)
    _x = ref_x[2]
    self.assertEqual(len(m.x), 3)
    self.assertIs(_x, m.x[2])
    self.assertEqual(value(_x), 2)
    self.assertEqual(value(m.x[2]), 2)
    self.assertEqual(value(ref_x[2]), 2)
    _y = ref_y[2]
    self.assertEqual(len(m.b), 3)
    self.assertIs(_y, m.b[2].y)
    self.assertEqual(value(_y), 20)
    self.assertEqual(value(ref_y[2]), 20)
    self.assertEqual(value(m.b[2].y), 20)