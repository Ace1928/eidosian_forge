import pickle
from collections import namedtuple
from datetime import datetime
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.base.indexed_component import IndexedComponent
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.common.log import LoggingIntercept
def test_find_wildcard_partial_exists(self):
    cuid = ComponentUID('b[*,*].c.a[**]')
    comp = cuid.find_component_on(self.m)
    self.assertIs(comp.ctype, Param)
    cList = list(comp.values())
    self.assertEqual(len(cList), 3)
    self.assertEqual(cList, list(self.m.b[1, '2'].c.a[:]))
    cuid = ComponentUID('b[*,*].c.a')
    comp = cuid.find_component_on(self.m)
    self.assertIs(comp.ctype, IndexedComponent)
    cList = list(comp.values())
    self.assertEqual(len(cList), 1)
    self.assertIs(cList[0], self.m.b[1, '2'].c.a)