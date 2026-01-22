import pickle
import unittest
from unittest import mock
from traits.api import HasTraits, Set, Str
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_set_object import TraitSet, TraitSetEvent
from traits.trait_types import _validate_int
def test_iand_does_not_call_validator(self):
    ts = TraitSet({1, 2, 3}, item_validator=self.validator)
    values = list(ts)
    python_set = set(ts)
    python_set &= set(values[:2])
    self.validator_args = None
    ts &= set(values[:2])
    self.assertEqual(ts, python_set)
    self.assertIsNone(self.validator_args)