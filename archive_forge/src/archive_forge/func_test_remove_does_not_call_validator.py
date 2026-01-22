import pickle
import unittest
from unittest import mock
from traits.api import HasTraits, Set, Str
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_set_object import TraitSet, TraitSetEvent
from traits.trait_types import _validate_int
def test_remove_does_not_call_validator(self):
    ts = TraitSet(item_validator=self.validator)
    ts.add('123')
    value, = ts
    self.validator_args = None
    ts.remove(value)
    self.assertIsNone(self.validator_args)