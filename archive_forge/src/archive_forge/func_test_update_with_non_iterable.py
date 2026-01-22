import pickle
import unittest
from unittest import mock
from traits.api import HasTraits, Set, Str
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_set_object import TraitSet, TraitSetEvent
from traits.trait_types import _validate_int
def test_update_with_non_iterable(self):
    python_set = set()
    with self.assertRaises(TypeError) as python_exc:
        python_set.update(None)
    ts = TraitSet()
    with self.assertRaises(TypeError) as trait_exc:
        ts.update(None)
    self.assertEqual(str(trait_exc.exception), str(python_exc.exception))