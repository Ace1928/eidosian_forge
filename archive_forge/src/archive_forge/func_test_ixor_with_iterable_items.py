import pickle
import unittest
from unittest import mock
from traits.api import HasTraits, Set, Str
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_set_object import TraitSet, TraitSetEvent
from traits.trait_types import _validate_int
def test_ixor_with_iterable_items(self):
    iterable = range(2)
    python_set = set([iterable])
    python_set ^= set([iterable])
    self.assertEqual(python_set, set())
    ts = TraitSet([iterable], item_validator=self.validator)
    self.validator_args = None
    ts ^= {iterable}
    self.assertEqual(ts, set())
    self.assertIsNone(self.validator_args)