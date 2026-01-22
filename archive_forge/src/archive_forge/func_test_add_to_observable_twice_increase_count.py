import unittest
from unittest import mock
import weakref
from traits.observation.exception_handling import (
from traits.observation.exceptions import NotifierNotFound
from traits.observation._trait_event_notifier import TraitEventNotifier
def test_add_to_observable_twice_increase_count(self):
    dummy = DummyObservable()

    def handler(event):
        pass
    notifier1 = create_notifier(handler=handler, target=_DUMMY_TARGET)
    notifier2 = create_notifier(handler=handler, target=_DUMMY_TARGET)
    notifier1.add_to(dummy)
    notifier2.add_to(dummy)
    self.assertEqual(dummy.notifiers, [notifier1])
    self.assertEqual(notifier1._ref_count, 2)