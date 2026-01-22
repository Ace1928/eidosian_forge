import unittest
from unittest import mock
import weakref
from traits.observation.exception_handling import (
from traits.observation.exceptions import NotifierNotFound
from traits.observation._trait_event_notifier import TraitEventNotifier
def test_init_and_call(self):
    handler = mock.Mock()

    def event_factory(*args, **kwargs):
        return 'Event'
    notifier = create_notifier(handler=handler, event_factory=event_factory)
    notifier(a=1, b=2)
    self.assertEqual(handler.call_count, 1)
    (args, _), = handler.call_args_list
    self.assertEqual(args, ('Event',))