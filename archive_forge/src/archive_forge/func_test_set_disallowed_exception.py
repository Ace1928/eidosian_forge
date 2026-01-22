import gc
import sys
import unittest
from traits.constants import DefaultValue
from traits.has_traits import (
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_errors import TraitError
from traits.trait_type import TraitType
from traits.trait_types import (
def test_set_disallowed_exception(self):

    class StrictDummy(HasStrictTraits):
        foo = Int
    with self.assertRaises(TraitError):
        StrictDummy(forbidden=53)
    with self.assertRaises(TraitError):
        StrictDummy(**{'forbidden': 53})
    a = StrictDummy()
    with self.assertRaises(TraitError):
        setattr(a, 'forbidden', 53)