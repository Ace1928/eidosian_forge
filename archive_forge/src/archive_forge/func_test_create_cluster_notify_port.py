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
@ddt.data({'notif_filters': (clusapi_def.NOTIFY_FILTER_AND_TYPE * 2)(), 'exp_notif_filters_len': 2}, {'notif_filters': clusapi_def.NOTIFY_FILTER_AND_TYPE(), 'notif_port_h': mock.sentinel.notif_port_h, 'notif_key': mock.sentinel.notif_key})
@ddt.unpack
def test_create_cluster_notify_port(self, notif_filters, exp_notif_filters_len=1, notif_port_h=None, notif_key=None):
    self._mock_ctypes()
    self._ctypes.Array = ctypes.Array
    self._clusapi_utils.create_cluster_notify_port_v2(mock.sentinel.cluster_handle, notif_filters, notif_port_h, notif_key)
    exp_notif_key_p = self._ctypes.byref(notif_key) if notif_key else None
    exp_notif_port_h = notif_port_h or w_const.INVALID_HANDLE_VALUE
    self._mock_run.assert_called_once_with(self._clusapi.CreateClusterNotifyPortV2, exp_notif_port_h, mock.sentinel.cluster_handle, self._ctypes.byref(notif_filters), self._ctypes.c_ulong(exp_notif_filters_len), exp_notif_key_p, **self._clusapi_utils._open_handle_check_flags)