import contextlib
import logging
import unittest
from traits import has_traits
from traits.api import (
from traits.adaptation.api import reset_global_adaptation_manager
from traits.interface_checker import InterfaceError
def test_adapts_to(self):
    ta = TraitsHolder()
    ta.list_adapts_to = object = Sample()
    self.assertEqual(ta.list_adapts_to, object)
    result = ta.list_adapts_to_.get_list()
    self.assertEqual(len(result), 3)
    for n in [1, 2, 3]:
        self.assertIn(n, result)
    self.assertIsInstance(ta.list_adapts_to_, SampleListAdapter)
    ta.foo_adapts_to = object = Sample()
    self.assertEqual(ta.foo_adapts_to, object)
    self.assertEqual(ta.foo_adapts_to_.get_foo(), 6)
    self.assertIsInstance(ta.foo_adapts_to_, SampleFooAdapter)
    ta.foo_plus_adapts_to = object = Sample(s1=5, s2=10, s3=15)
    self.assertEqual(ta.foo_plus_adapts_to, object)
    self.assertEqual(ta.foo_plus_adapts_to_.get_foo(), 30)
    self.assertEqual(ta.foo_plus_adapts_to_.get_foo_plus(), 31)
    self.assertIsInstance(ta.foo_plus_adapts_to_, FooPlusAdapter)