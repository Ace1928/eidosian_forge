import ctypes
from unittest import mock
import ddt
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.compute import _clusapi_utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi.libs import clusapi as clusapi_def
from os_win.utils.winapi import wintypes
def test_destroy_cluster_group(self):
    self._clusapi_utils.destroy_cluster_group(mock.sentinel.group_handle)
    self._mock_run.assert_called_once_with(self._clusapi.DestroyClusterGroup, mock.sentinel.group_handle)