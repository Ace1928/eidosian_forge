import ctypes
from unittest import mock
import ddt
from six.moves import queue
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.compute import clusterutils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi.libs import clusapi as clusapi_def
from os_win.utils.winapi import wintypes
def test_signal_stopped(self):
    self._listener._signal_stopped()
    self.assertFalse(self._listener._running)
    self.assertIsNone(self._listener._event_queue.get(block=False))