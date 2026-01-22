import pickle
from collections import namedtuple
from datetime import datetime
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.base.indexed_component import IndexedComponent
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.common.log import LoggingIntercept
def test_printers_2(self):
    cuid = ComponentUID('b:$1,2.c.a:#3')
    s = "b['1',2].c.a[3]"
    r1 = 'b:$1,#2.c.a:#3'
    r2 = "b['1',2].c.a[3]"
    self.assertEqual(str(cuid), s)
    self.assertEqual(repr(cuid), r2)
    self.assertEqual(cuid.get_repr(1), r1)
    self.assertEqual(cuid.get_repr(2), r2)