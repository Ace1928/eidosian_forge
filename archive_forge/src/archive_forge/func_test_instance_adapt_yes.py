import contextlib
import logging
import unittest
from traits import has_traits
from traits.api import (
from traits.adaptation.api import reset_global_adaptation_manager
from traits.interface_checker import InterfaceError
def test_instance_adapt_yes(self):
    ta = TraitsHolder()
    ta.a_yes = SampleAverage()
    self.assertEqual(ta.a_yes.get_average(), 200.0)
    self.assertIsInstance(ta.a_yes, SampleAverage)
    self.assertFalse(hasattr(ta, 'a_yes_'))
    ta.a_yes = SampleList()
    self.assertEqual(ta.a_yes.get_average(), 20.0)
    self.assertIsInstance(ta.a_yes, ListAverageAdapter)
    self.assertFalse(hasattr(ta, 'a_yes_'))
    ta.a_yes = Sample()
    self.assertEqual(ta.a_yes.get_average(), 2.0)
    self.assertIsInstance(ta.a_yes, ListAverageAdapter)
    self.assertFalse(hasattr(ta, 'a_yes_'))
    self.assertRaises(TraitError, ta.trait_set, a_yes=SampleBad())