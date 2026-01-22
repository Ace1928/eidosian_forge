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
def test_process_state_change_event(self):
    fake_state = constants.CLUSTER_GROUP_ONLINE
    event_buff = ctypes.c_ulong(fake_state)
    notif_key = self._listener._NOTIF_KEY_GROUP_STATE
    event = self._get_fake_event(notif_key=notif_key, buff=ctypes.byref(event_buff), buff_sz=ctypes.sizeof(event_buff))
    exp_proc_evt = self._get_exp_processed_event(event, state=fake_state)
    proc_evt = self._listener._process_event(event)
    self.assertEqual(exp_proc_evt, proc_evt)