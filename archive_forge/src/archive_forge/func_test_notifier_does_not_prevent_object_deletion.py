import unittest
from unittest import mock
import weakref
from traits.observation.exception_handling import (
from traits.observation.exceptions import NotifierNotFound
from traits.observation._trait_event_notifier import TraitEventNotifier
def test_notifier_does_not_prevent_object_deletion(self):
    target = DummyObservable()
    target.internal_object = DummyObservable()
    target_ref = weakref.ref(target)
    notifier = create_notifier(target=target)
    notifier.add_to(target.internal_object)
    del target
    self.assertIsNone(target_ref())