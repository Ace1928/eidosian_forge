import pickle
from collections import namedtuple
from datetime import datetime
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.base.indexed_component import IndexedComponent
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.common.log import LoggingIntercept
def test_genFromComponent_nameBuffer(self):
    buf = {}
    cuid = ComponentUID(self.m.b[1, '2'].c.a, cuid_buffer=buf)
    self.assertEqual(cuid._cids, (('b', (1, '2')), ('c', tuple()), ('a', ())))
    self.assertEqual(len(buf), 9)
    for s1 in self.m.s:
        for s2 in self.m.s:
            _id = id(self.m.b[s1, s2])
            self.assertIn(_id, buf)
            self.assertEqual(buf[_id], ('b', (s1, s2)))