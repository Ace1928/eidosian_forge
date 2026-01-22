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
@mock.patch.object(clusterutils.ClusterUtils, '_get_cluster_nodes')
def get_cluster_node_name(self, mock_get_nodes):
    fake_node = dict(id=mock.sentinel.vm_id, name=mock.sentinel.vm_name)
    mock_get_nodes.return_value([fake_node])
    self.assertEqual(mock.sentinel.vm_name, self._clusterutils.get_cluster_node_name(mock.sentinel.vm_id))
    self.assertRaises(exceptions.NotFound, self._clusterutils.get_cluster_node_name(mock.sentinel.missing_id))