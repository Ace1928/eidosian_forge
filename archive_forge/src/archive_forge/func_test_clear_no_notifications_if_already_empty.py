import pickle
import unittest
from unittest import mock
from traits.api import HasTraits, Set, Str
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_set_object import TraitSet, TraitSetEvent
from traits.trait_types import _validate_int
def test_clear_no_notifications_if_already_empty(self):
    notifier = mock.Mock()
    ts = TraitSet(notifiers=[notifier])
    ts.clear()
    notifier.assert_not_called()