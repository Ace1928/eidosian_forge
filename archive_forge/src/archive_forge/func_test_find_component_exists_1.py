import pickle
from collections import namedtuple
from datetime import datetime
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.base.indexed_component import IndexedComponent
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.common.log import LoggingIntercept
def test_find_component_exists_1(self):
    ref = self.m.b[1, '2'].c.a
    cuid = ComponentUID(ref)
    self.assertTrue(cuid.find_component_on(self.m) is ref)