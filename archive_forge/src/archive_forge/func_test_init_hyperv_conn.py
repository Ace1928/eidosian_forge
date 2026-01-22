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
def test_init_hyperv_conn(self):
    fake_cluster_name = 'fake_cluster'
    mock_cluster = mock.MagicMock()
    mock_cluster.path_.return_value = '\\\\%s\\root' % fake_cluster_name
    mock_conn = mock.MagicMock()
    mock_conn.MSCluster_Cluster.return_value = [mock_cluster]
    self._clusterutils._get_wmi_conn = mock.MagicMock()
    self._clusterutils._get_wmi_conn.return_value = mock_conn
    self._clusterutils._init_hyperv_conn('fake_host', timeout=1)