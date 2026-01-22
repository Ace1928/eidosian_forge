import unittest
from unittest import mock
import weakref
from traits.observation.exception_handling import (
from traits.observation.exceptions import NotifierNotFound
from traits.observation._trait_event_notifier import TraitEventNotifier
def test_remove_from_differentiate_not_equal_notifier(self):
    dummy = DummyObservable()
    notifier1 = create_notifier(handler=mock.Mock())
    notifier2 = create_notifier(handler=mock.Mock())
    notifier1.add_to(dummy)
    notifier2.add_to(dummy)
    notifier2.remove_from(dummy)
    self.assertEqual(dummy.notifiers, [notifier1])