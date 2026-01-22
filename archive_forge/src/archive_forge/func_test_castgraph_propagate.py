import itertools
from numba.core import types
from numba.core.typeconv.typeconv import TypeManager, TypeCastingRules
from numba.core.typeconv import rules
from numba.core.typeconv import castgraph, Conversion
import unittest
def test_castgraph_propagate(self):
    saved = []

    def callback(src, dst, rel):
        saved.append((src, dst, rel))
    tg = castgraph.TypeGraph(callback)
    i32 = types.int32
    i64 = types.int64
    f64 = types.float64
    f32 = types.float32
    tg.insert_rule(i32, i64, Conversion.promote)
    tg.insert_rule(i64, i32, Conversion.unsafe)
    saved.append(None)
    tg.insert_rule(i32, f64, Conversion.safe)
    tg.insert_rule(f64, i32, Conversion.unsafe)
    saved.append(None)
    tg.insert_rule(f32, f64, Conversion.promote)
    tg.insert_rule(f64, f32, Conversion.unsafe)
    self.assertIn((i32, i64, Conversion.promote), saved[0:2])
    self.assertIn((i64, i32, Conversion.unsafe), saved[0:2])
    self.assertIs(saved[2], None)
    self.assertIn((i32, f64, Conversion.safe), saved[3:7])
    self.assertIn((f64, i32, Conversion.unsafe), saved[3:7])
    self.assertIn((i64, f64, Conversion.unsafe), saved[3:7])
    self.assertIn((i64, f64, Conversion.unsafe), saved[3:7])
    self.assertIs(saved[7], None)
    self.assertIn((f32, f64, Conversion.promote), saved[8:14])
    self.assertIn((f64, f32, Conversion.unsafe), saved[8:14])
    self.assertIn((f32, i32, Conversion.unsafe), saved[8:14])
    self.assertIn((i32, f32, Conversion.unsafe), saved[8:14])
    self.assertIn((f32, i64, Conversion.unsafe), saved[8:14])
    self.assertIn((i64, f32, Conversion.unsafe), saved[8:14])
    self.assertEqual(len(saved[14:]), 0)