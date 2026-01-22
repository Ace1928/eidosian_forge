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
def test_get_notif_key_dw(self):
    fake_notif_key = 1
    notif_key_dw = self._listener._get_notif_key_dw(fake_notif_key)
    self.assertIsInstance(notif_key_dw, ctypes.c_ulong)
    self.assertEqual(fake_notif_key, notif_key_dw.value)
    self.assertEqual(notif_key_dw, self._listener._get_notif_key_dw(fake_notif_key))