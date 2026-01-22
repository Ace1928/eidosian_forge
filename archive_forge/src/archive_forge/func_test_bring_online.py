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
def test_bring_online(self):
    self._clusterutils.bring_online(mock.sentinel.vm_name)
    self._cmgr.open_cluster_group.assert_called_once_with(mock.sentinel.vm_name)
    self._clusapi.online_cluster_group.assert_called_once_with(self._cmgr_val(self._cmgr.open_cluster_group))