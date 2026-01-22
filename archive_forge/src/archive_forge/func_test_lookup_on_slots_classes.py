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
def test_lookup_on_slots_classes(self):
    sga = inspection.getattr_safe
    s = SlotsSubclass()
    self.assertIsInstance(sga(Slots, 's1'), member_descriptor)
    self.assertIsInstance(sga(SlotsSubclass, 's1'), member_descriptor)
    self.assertIsInstance(sga(SlotsSubclass, 's4'), property)
    self.assertIsInstance(sga(s, 's4'), property)
    self.assertEqual(inspection.hasattr_safe(s, 's1'), False)
    self.assertEqual(inspection.hasattr_safe(s, 's4'), True)