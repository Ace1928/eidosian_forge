import unittest
from unittest import mock
import weakref
from traits.observation.exception_handling import (
from traits.observation.exceptions import NotifierNotFound
from traits.observation._trait_event_notifier import TraitEventNotifier
def test_not_equal_if_dispatcher_different(self):
    handler = mock.Mock()
    target = mock.Mock()
    dispatcher1 = mock.Mock()
    dispatcher2 = mock.Mock()
    notifier1 = create_notifier(handler=handler, target=target, dispatcher=dispatcher1)
    notifier2 = create_notifier(handler=handler, target=target, dispatcher=dispatcher2)
    self.assertFalse(notifier1.equals(notifier2), 'Expected the notifiers to be different because the dispatchers do not compare equally.')
    self.assertFalse(notifier2.equals(notifier1), 'Expected the notifiers to be different because the dispatchers do not compare equally.')