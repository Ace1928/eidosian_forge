import unittest
from unittest import mock
import weakref
from traits.observation.exception_handling import (
from traits.observation.exceptions import NotifierNotFound
from traits.observation._trait_event_notifier import TraitEventNotifier
def test_add_to_multiple_observables(self):
    dummy1 = DummyObservable()
    dummy2 = DummyObservable()
    notifier = create_notifier()
    notifier.add_to(dummy1)
    with self.assertRaises(RuntimeError) as exception_context:
        notifier.add_to(dummy2)
    self.assertEqual(str(exception_context.exception), 'Sharing notifiers across observables is unexpected.')