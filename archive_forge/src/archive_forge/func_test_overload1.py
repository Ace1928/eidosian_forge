import itertools
from numba.core import types
from numba.core.typeconv.typeconv import TypeManager, TypeCastingRules
from numba.core.typeconv import rules
from numba.core.typeconv import castgraph, Conversion
import unittest
def test_overload1(self):
    tm = rules.default_type_manager
    i32 = types.int32
    i64 = types.int64
    sig = (i64, i32, i32)
    ovs = [(i32, i32, i32), (i64, i64, i64)]
    self.assertEqual(tm.select_overload(sig, ovs, True, False), 1)
    self.assertEqual(tm.select_overload(sig, ovs, False, False), 1)