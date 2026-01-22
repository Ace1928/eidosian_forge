import threading
from unittest import mock
import warnings
import eventlet
from eventlet import greenthread
from oslotest import base as test_base
from oslo_utils import eventletutils
def test_event_no_timeout(self):
    event = eventletutils.EventletEvent()

    def thread_a():
        self.assertTrue(event.wait())
    a = greenthread.spawn(thread_a)
    with eventlet.timeout.Timeout(0.5, False):
        a.wait()
        self.fail('wait() timed out')