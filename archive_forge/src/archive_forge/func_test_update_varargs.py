import pickle
import unittest
from unittest import mock
from traits.api import HasTraits, Set, Str
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_set_object import TraitSet, TraitSetEvent
from traits.trait_types import _validate_int
def test_update_varargs(self):
    ts = TraitSet(notifiers=[self.notification_handler])
    ts.update({1, 2}, {3, 4})
    self.assertEqual(self.added, {1, 2, 3, 4})
    self.assertEqual(self.removed, set())