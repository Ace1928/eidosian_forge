import itertools
from numba.core import types
from numba.core.typeconv.typeconv import TypeManager, TypeCastingRules
from numba.core.typeconv import rules
from numba.core.typeconv import castgraph, Conversion
import unittest
def test_overload4(self):
    tm = rules.default_type_manager
    i16 = types.int16
    i32 = types.int32
    i64 = types.int64
    f16 = types.float16
    f32 = types.float32
    sig = (i16, f16, f16)
    ovs = [(f16, f32, f16), (f32, i32, f16)]
    self.assertEqual(tm.select_overload(sig, ovs, allow_unsafe=True, exact_match_required=False), 0)