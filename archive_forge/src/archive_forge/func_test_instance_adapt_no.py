import contextlib
import logging
import unittest
from traits import has_traits
from traits.api import (
from traits.adaptation.api import reset_global_adaptation_manager
from traits.interface_checker import InterfaceError
def test_instance_adapt_no(self):
    ta = TraitsHolder()
    try:
        ta.a_no = SampleAverage()
    except TraitError:
        self.fail('Setting instance of interface should not require adaptation')
    self.assertRaises(TraitError, ta.trait_set, a_no=SampleList())
    self.assertRaises(TraitError, ta.trait_set, a_no=Sample())
    self.assertRaises(TraitError, ta.trait_set, a_no=SampleBad())