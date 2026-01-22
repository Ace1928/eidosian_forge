import threading
from unittest import mock
import warnings
import eventlet
from eventlet import greenthread
from oslotest import base as test_base
from oslo_utils import eventletutils
def test_event_clear_timeout(self):
    event = eventletutils.EventletEvent()

    def thread_a():
        self.assertFalse(event.wait(0.5))
    a = greenthread.spawn(thread_a)

    def thread_b():
        eventlet.sleep(0.1)
        event.clear()
        eventlet.sleep(0.1)
        event.clear()
        a.wait()
    b = greenthread.spawn(thread_b)
    with eventlet.timeout.Timeout(0.7):
        b.wait()