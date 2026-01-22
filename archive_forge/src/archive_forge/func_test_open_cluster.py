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
@ddt.data(None, mock.sentinel.cluster_name)
def test_open_cluster(self, cluster_name):
    with self._cmgr.open_cluster(cluster_name) as f:
        self._clusapi_utils.open_cluster.assert_called_once_with(cluster_name)
        self.assertEqual(f, self._clusapi_utils.open_cluster.return_value)
    self._clusapi_utils.close_cluster.assert_called_once_with(self._clusapi_utils.open_cluster.return_value)