import unittest
from unittest import mock
import weakref
from traits.observation.exception_handling import (
from traits.observation.exceptions import NotifierNotFound
from traits.observation._trait_event_notifier import TraitEventNotifier
def test_method_as_handler_does_not_prevent_garbage_collect(self):
    dummy = DummyObservable()
    dummy.internal_object = DummyObservable()
    dummy_ref = weakref.ref(dummy)
    notifier = create_notifier(handler=dummy.handler)
    notifier.add_to(dummy.internal_object)
    del dummy
    self.assertIsNone(dummy_ref())