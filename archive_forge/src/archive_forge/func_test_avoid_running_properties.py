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
def test_avoid_running_properties(self):
    p = Property()
    self.assertEqual(inspection.getattr_safe(p, 'prop'), Property.prop)
    self.assertEqual(inspection.hasattr_safe(p, 'prop'), True)