import pickle
import unittest
from unittest import mock
from traits.api import HasTraits, Set, Str
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_set_object import TraitSet, TraitSetEvent
from traits.trait_types import _validate_int
def test_add_no_notification_for_no_op(self):
    notifier = mock.Mock()
    ts = TraitSet({1, 2}, notifiers=[notifier])
    ts.add(1)
    notifier.assert_not_called()