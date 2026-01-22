import pickle
from collections import namedtuple
from datetime import datetime
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.base.indexed_component import IndexedComponent
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.common.log import LoggingIntercept
def test_cuid_from_slice_1(self):
    """
        These are slices over a single level of the hierarchy.
        """
    m = self._slice_model()
    _slice = m.b[:]
    cuid_str = ComponentUID('b[*]')
    cuid = ComponentUID(_slice)
    self.assertEqual(cuid, cuid_str)
    _slice = m.b.b1[:]
    cuid_str = ComponentUID('b.b1[*]')
    cuid = ComponentUID(_slice)
    self.assertEqual(cuid, cuid_str)
    _slice = m.b.b1[...]
    cuid_str = ComponentUID('b.b1[**]')
    cuid = ComponentUID(_slice)
    self.assertEqual(cuid, cuid_str)
    _slice = m.b.b2[:, 'a']
    cuid_str = ComponentUID('b.b2[*,a]')
    cuid = ComponentUID(_slice)
    self.assertEqual(cuid, cuid_str)
    self.assertEqual(str(cuid), str(cuid_str))
    self.assertListSameComponents(m, cuid, cuid_str)
    _slice = m.b.b2[...]
    cuid_str = ComponentUID('b.b2[**]')
    cuid = ComponentUID(_slice)
    self.assertEqual(cuid, cuid_str)
    _slice = m.b.b3[1.1, :, 2]
    cuid_str = ComponentUID('b.b3[1.1,*,2]')
    cuid = ComponentUID(_slice)
    self.assertEqual(cuid, cuid_str)
    self.assertEqual(str(cuid), str(cuid_str))
    self.assertListSameComponents(m, cuid, cuid_str)
    _slice = m.b.b3[:, :, 'b']
    cuid_str = ComponentUID('b.b3[*,*,b]')
    cuid = ComponentUID(_slice)
    self.assertEqual(cuid, cuid_str)
    self.assertEqual(str(cuid), str(cuid_str))
    self.assertListSameComponents(m, cuid, cuid_str)
    _slice = m.b.b3[1.1, ...]
    cuid_str = ComponentUID('b.b3[1.1,**]')
    cuid = ComponentUID(_slice)
    self.assertEqual(cuid, cuid_str)
    self.assertListSameComponents(m, cuid, cuid_str)
    _slice = m.b.b3[...]
    cuid_str = ComponentUID('b.b3[**]')
    cuid = ComponentUID(_slice)
    self.assertEqual(cuid, cuid_str)
    self.assertEqual(str(cuid), str(cuid_str))
    _slice = m.b.bn['a', :, :, 'a', 1]
    cuid_str = ComponentUID('b.bn[a,*,*,a,1]')
    cuid = ComponentUID(_slice)
    self.assertEqual(cuid, cuid_str)
    self.assertEqual(str(cuid), str(cuid_str))
    self.assertListSameComponents(m, cuid, cuid_str)
    _slice = m.b.bn['a', 'c', 3, :, :]
    cuid_str = ComponentUID('b.bn[a,c,3,*,*]')
    cuid = ComponentUID(_slice)
    self.assertEqual(cuid, cuid_str)
    self.assertEqual(str(cuid), str(cuid_str))
    self.assertListSameComponents(m, cuid, cuid_str)
    _slice = m.b.bn[...]
    cuid_str = ComponentUID('b.bn[**]')
    cuid = ComponentUID(_slice)
    self.assertEqual(cuid, cuid_str)