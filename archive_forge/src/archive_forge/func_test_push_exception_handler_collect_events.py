import unittest
from unittest import mock
from traits.observation.exception_handling import (
def test_push_exception_handler_collect_events(self):
    events = []

    def handler(event):
        events.append(event)
    stack = ObserverExceptionHandlerStack()
    stack.push_exception_handler(handler=handler)
    try:
        raise ZeroDivisionError()
    except Exception:
        stack.handle_exception('Event')
    self.assertEqual(events, ['Event'])