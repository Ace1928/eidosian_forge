import itertools
from numba.core import types
from numba.core.typeconv.typeconv import TypeManager, TypeCastingRules
from numba.core.typeconv import rules
from numba.core.typeconv import castgraph, Conversion
import unittest
def test_typeconv(self):
    tm = TypeManager()
    i32 = types.int32
    i64 = types.int64
    f32 = types.float32
    tm.set_promote(i32, i64)
    tm.set_unsafe_convert(i32, f32)
    sig = (i32, f32)
    ovs = [(i32, i32), (f32, f32), (i64, i64)]
    sel = tm.select_overload(sig, ovs, True, False)
    self.assertEqual(sel, 1)
    with self.assertRaises(TypeError):
        sel = tm.select_overload(sig, ovs, False, False)