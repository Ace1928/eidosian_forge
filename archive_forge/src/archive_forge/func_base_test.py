import itertools
from numba.core import types
from numba.core.typeconv.typeconv import TypeManager, TypeCastingRules
from numba.core.typeconv import rules
from numba.core.typeconv import castgraph, Conversion
import unittest
def base_test():
    self.assertEqual(tm.check_compatible(i32, i64), Conversion.promote)
    self.assertEqual(tm.check_compatible(i32, f64), Conversion.safe)
    self.assertEqual(tm.check_compatible(f16, f32), Conversion.promote)
    self.assertEqual(tm.check_compatible(f32, f64), Conversion.promote)
    self.assertEqual(tm.check_compatible(i64, i32), Conversion.unsafe)
    self.assertEqual(tm.check_compatible(f64, i32), Conversion.unsafe)
    self.assertEqual(tm.check_compatible(f64, f32), Conversion.unsafe)
    self.assertEqual(tm.check_compatible(i64, f64), Conversion.unsafe)
    self.assertEqual(tm.check_compatible(f64, i64), Conversion.unsafe)
    self.assertEqual(tm.check_compatible(i64, f32), Conversion.unsafe)
    self.assertEqual(tm.check_compatible(i32, f32), Conversion.unsafe)
    self.assertEqual(tm.check_compatible(f32, i32), Conversion.unsafe)
    self.assertEqual(tm.check_compatible(i16, f16), Conversion.unsafe)
    self.assertEqual(tm.check_compatible(f16, i16), Conversion.unsafe)