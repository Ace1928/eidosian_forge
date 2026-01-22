import sys
import unittest.mock
import warnings
import weakref
from traits.api import HasTraits
from traits.constants import (
from traits.ctrait import CTrait
from traits.trait_errors import TraitError
from traits.trait_types import Any, Int, List
def test_failed_attribute_access(self):
    non_dunder_names = ['non_existent', '__non_existent', 'non_existent__', '_non_existent_', '__a__b_', '_']
    dunder_names = ['__package__', '__a__', '____', '___', '__']
    ctrait = CTrait(0)
    for name in non_dunder_names:
        with self.subTest(name=name):
            self.assertIsNone(getattr(ctrait, name))
    for name in dunder_names:
        with self.subTest(name=name):
            with self.assertRaises(AttributeError):
                getattr(ctrait, name)