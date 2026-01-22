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
def test_referent(self):
    m = ConcreteModel()
    m.v0 = Var()
    m.v2 = Var([1, 2, 3], ['a', 'b'])
    varlist = [m.v2[1, 'a'], m.v2[1, 'b']]
    vardict = {0: m.v0, 1: m.v2[1, 'a'], 2: m.v2[2, 'a'], 3: m.v2[3, 'a']}
    scalar_ref = Reference(m.v0)
    self.assertIs(scalar_ref.referent, m.v0)
    sliced_ref = Reference(m.v2[:, 'a'])
    referent = sliced_ref.referent
    self.assertIs(type(referent), IndexedComponent_slice)
    self.assertEqual(len(referent._call_stack), 1)
    call, info = referent._call_stack[0]
    self.assertEqual(call, IndexedComponent_slice.slice_info)
    self.assertIs(info[0], m.v2)
    self.assertEqual(info[1], {1: 'a'})
    self.assertEqual(info[2], {0: slice(None)})
    self.assertIs(info[3], None)
    list_ref = Reference(varlist)
    self.assertIs(list_ref.referent, varlist)
    dict_ref = Reference(vardict)
    self.assertIs(dict_ref.referent, vardict)