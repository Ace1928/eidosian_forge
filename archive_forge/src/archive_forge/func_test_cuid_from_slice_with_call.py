import pickle
from collections import namedtuple
from datetime import datetime
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.base.indexed_component import IndexedComponent
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.common.log import LoggingIntercept
def test_cuid_from_slice_with_call(self):
    m = self._slice_model()
    _slice = m.b.component('b2')[:, 'a'].v2[1, :]
    cuid = ComponentUID(_slice)
    cuid_str = ComponentUID('b.b2[*,a].v2[1,*]')
    self.assertEqual(cuid, cuid_str)
    self.assertEqual(str(cuid), str(cuid_str))
    self.assertListSameComponents(m, cuid, cuid_str)
    _slice = m.b.find_component('b2')[:, 'a'].v2[1, :]
    cuid = ComponentUID(_slice)
    cuid_str = ComponentUID('b.b2[*,a].v2[1,*]')
    self.assertEqual(cuid, cuid_str)
    self.assertEqual(str(cuid), str(cuid_str))
    self.assertListSameComponents(m, cuid, cuid_str)
    _slice = m.b[:].component('b2')
    cuid = ComponentUID(_slice)
    cuid_str = ComponentUID('b[*].b2')
    self.assertEqual(str(cuid), str(cuid_str))
    self.assertListSameComponents(m, cuid, cuid_str)
    _slice = m.b[:].component('b2', 'b1')
    with self.assertRaisesRegex(ValueError, '.*multiple arguments.*'):
        cuid = ComponentUID(_slice)
    _slice = IndexedComponent_slice(m.b[:].fix, (IndexedComponent_slice.call, ('fix',), {}))
    with self.assertRaisesRegex(ValueError, "Cannot create a CUID from a slice with a call to any method other than 'component': got 'fix'\\."):
        cuid = ComponentUID(_slice)
    _slice = IndexedComponent_slice(m.b[:].component('v'), (IndexedComponent_slice.call, ('fix',), {}))
    with self.assertRaisesRegex(ValueError, "Cannot create a CUID with a __call__ of anything other than a 'component' attribute"):
        cuid = ComponentUID(_slice)
    _slice = m.b[:].component('b2', kwd=None)
    with self.assertRaisesRegex(ValueError, '.*call that contains keywords.*'):
        cuid = ComponentUID(_slice)
    _slice = m.b.b2[:, 'a'].component('vn')[:, 'c', 3, :, :]
    cuid = ComponentUID(_slice)
    cuid_str = ComponentUID('b.b2[*,a].vn[*,c,3,*,*]')
    self.assertEqual(cuid, cuid_str)
    self.assertEqual(str(cuid), str(cuid_str))
    self.assertListSameComponents(m, cuid, cuid_str)
    _slice = m.b.b2[1, 'a'].component('vn')[:, 'c', 3, :, :]
    cuid = ComponentUID(_slice)
    cuid_str = ComponentUID('b.b2[1,a].vn[*,c,3,*,*]')
    self.assertEqual(cuid, cuid_str)
    self.assertEqual(str(cuid), str(cuid_str))
    self.assertListSameComponents(m, cuid, cuid_str)
    _slice = m.b.b2[...].component('vn')[:, 'c', 3, :, :]
    cuid = ComponentUID(_slice)
    cuid_str = ComponentUID('b.b2[**].vn[*,c,3,*,*]')
    self.assertEqual(cuid, cuid_str)
    self.assertEqual(str(cuid), str(cuid_str))
    self.assertListSameComponents(m, cuid, cuid_str)
    _slice = m.b.b2[:, 'a'].component('vn')[...]
    cuid = ComponentUID(_slice)
    cuid_str = ComponentUID('b.b2[*,a].vn[**]')
    self.assertEqual(cuid, cuid_str)
    self.assertEqual(str(cuid), str(cuid_str))
    self.assertListSameComponents(m, cuid, cuid_str)