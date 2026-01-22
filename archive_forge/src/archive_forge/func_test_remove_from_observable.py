import unittest
from unittest import mock
import weakref
from traits.observation.exception_handling import (
from traits.observation.exceptions import NotifierNotFound
from traits.observation._trait_event_notifier import TraitEventNotifier
def test_remove_from_observable(self):
    dummy = DummyObservable()

    def handler(event):
        pass
    notifier1 = create_notifier(handler=handler, target=_DUMMY_TARGET)
    notifier2 = create_notifier(handler=handler, target=_DUMMY_TARGET)
    notifier1.add_to(dummy)
    self.assertEqual(dummy.notifiers, [notifier1])
    notifier2.remove_from(dummy)
    self.assertEqual(dummy.notifiers, [])