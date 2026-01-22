import inspect
import os
import sys
import unittest
from collections.abc import Sequence
from typing import List
from bpython import inspection
from bpython.test.fodder import encoding_ascii
from bpython.test.fodder import encoding_latin1
from bpython.test.fodder import encoding_utf8
def test_lookup_on_object(self):
    a = A()
    a.x = 1
    self.assertEqual(inspection.getattr_safe(a, 'x'), 1)
    self.assertEqual(inspection.getattr_safe(a, 'a'), 'a')
    b = B()
    b.y = 2
    self.assertEqual(inspection.getattr_safe(b, 'y'), 2)
    self.assertEqual(inspection.getattr_safe(b, 'a'), 'a')
    self.assertEqual(inspection.getattr_safe(b, 'b'), 'b')
    self.assertEqual(inspection.hasattr_safe(b, 'y'), True)
    self.assertEqual(inspection.hasattr_safe(b, 'b'), True)