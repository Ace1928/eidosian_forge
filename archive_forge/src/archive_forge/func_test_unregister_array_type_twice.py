import re
import unittest
from wsme import exc
from wsme import types
def test_unregister_array_type_twice(self):

    class TempType(object):
        pass
    t = [TempType]
    types.registry.register(t)
    self.assertNotEqual(types.registry.array_types, set())
    types.registry._unregister(t)
    types.registry._unregister(t)
    self.assertEqual(types.registry.array_types, set())