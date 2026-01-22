from itertools import product, permutations
from collections import defaultdict
import unittest
from numba.core.base import OverloadSelector
from numba.core.registry import cpu_target
from numba.core.imputils import builtin_registry, RegistryLoader
from numba.core import types
from numba.core.errors import NumbaNotImplementedError, NumbaTypeError
def test_ambiguous_detection(self):
    os = OverloadSelector()
    os.append(1, (types.Any, types.Boolean))
    os.append(2, (types.Integer, types.Boolean))
    self.assertEqual(os.find((types.boolean, types.boolean)), 1)
    with self.assertRaises(NumbaNotImplementedError) as raises:
        os.find((types.boolean, types.int32))
    os.append(3, (types.Any, types.Any))
    self.assertEqual(os.find((types.boolean, types.int32)), 3)
    self.assertEqual(os.find((types.boolean, types.boolean)), 1)
    os.append(4, (types.Boolean, types.Any))
    with self.assertRaises(NumbaTypeError) as raises:
        os.find((types.boolean, types.boolean))
    self.assertIn('2 ambiguous signatures', str(raises.exception))
    os.append(5, (types.boolean, types.boolean))
    self.assertEqual(os.find((types.boolean, types.boolean)), 5)