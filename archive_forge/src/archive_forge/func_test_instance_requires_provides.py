import contextlib
import logging
import unittest
from traits import has_traits
from traits.api import (
from traits.adaptation.api import reset_global_adaptation_manager
from traits.interface_checker import InterfaceError
def test_instance_requires_provides(self):
    ta = TraitsHolder()
    provider = UndeclaredAverageProvider()
    with self.assertRaises(TraitError):
        ta.a_no = provider