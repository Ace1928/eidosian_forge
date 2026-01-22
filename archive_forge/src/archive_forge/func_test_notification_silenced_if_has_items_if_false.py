import pickle
import unittest
from unittest import mock
from traits.api import HasTraits, Set, Str
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_set_object import TraitSet, TraitSetEvent
from traits.trait_types import _validate_int
def test_notification_silenced_if_has_items_if_false(self):

    class Foo(HasTraits):
        values = Set(items=False)
    foo = Foo(values=set())
    notifier = mock.Mock()
    foo.on_trait_change(lambda: notifier(), 'values_items')
    foo.values.add(1)
    notifier.assert_not_called()