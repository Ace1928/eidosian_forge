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
def test_ctype_detection(self):
    m = ConcreteModel()
    m.js = Set(initialize=[1, (2, 3)], dimen=None)
    m.b = Block([1, 2])
    m.b[1].x = Var(m.js)
    m.b[1].y = Var()
    m.b[1].z = Var([1, 2])
    m.b[2].x = Param(initialize=0)
    m.b[2].y = Var()
    m.b[2].z = Var([1, 2])
    m.x = Reference(m.b[:].x[...])
    self.assertIs(type(m.x), IndexedComponent)
    m.y = Reference(m.b[:].y[...])
    self.assertIs(type(m.y), IndexedVar)
    self.assertIs(m.y.ctype, Var)
    m.y1 = Reference(m.b[:].y[...], ctype=None)
    self.assertIs(type(m.y1), IndexedComponent)
    self.assertIs(m.y1.ctype, IndexedComponent)
    m.z = Reference(m.b[:].z)
    self.assertIs(type(m.z), IndexedComponent)
    self.assertIs(m.z.ctype, IndexedComponent)