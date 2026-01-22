import time
from unittest import mock
import eventlet
from eventlet.green import threading as greenthreading
from oslotest import base as test_base
from oslo_service import fixture
from oslo_service import loopingcall
def test_eventlet_clock(self):
    hub = eventlet.hubs.get_hub()
    self.assertEqual(time.monotonic, hub.clock)