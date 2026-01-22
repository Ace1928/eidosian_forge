import contextlib
import unittest
from unittest import mock
import eventlet
from eventlet import debug as eventlet_debug
from eventlet import greenpool
from oslo_log import pipe_mutex
def test_wrong_releaser(self):
    self.mutex.acquire()
    with quiet_eventlet_exceptions():
        self.assertRaises(RuntimeError, eventlet.spawn(self.mutex.release).wait)