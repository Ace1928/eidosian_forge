import pickle
import unittest
from unittest import mock
from traits.api import HasTraits, Set, Str
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_set_object import TraitSet, TraitSetEvent
from traits.trait_types import _validate_int
def test_init_with_no_input(self):
    ts = TraitSet()
    self.assertEqual(ts, set())
    self.assertIs(ts.item_validator, _validate_everything)
    self.assertEqual(ts.notifiers, [])