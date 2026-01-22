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
def test_lookup_on_overridden_methods(self):
    sga = inspection.getattr_safe
    self.assertEqual(sga(OverriddenGetattr(), 'a'), 1)
    self.assertEqual(sga(OverriddenGetattribute(), 'a'), 1)
    self.assertEqual(sga(OverriddenMRO(), 'a'), 1)
    with self.assertRaises(AttributeError):
        sga(OverriddenGetattr(), 'b')
    with self.assertRaises(AttributeError):
        sga(OverriddenGetattribute(), 'b')
    with self.assertRaises(AttributeError):
        sga(OverriddenMRO(), 'b')
    self.assertEqual(inspection.hasattr_safe(OverriddenGetattr(), 'b'), False)
    self.assertEqual(inspection.hasattr_safe(OverriddenGetattribute(), 'b'), False)
    self.assertEqual(inspection.hasattr_safe(OverriddenMRO(), 'b'), False)