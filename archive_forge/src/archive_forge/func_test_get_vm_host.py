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
@mock.patch.object(clusterutils.ClusterUtils, '_get_cluster_group_state')
def test_get_vm_host(self, mock_get_state):
    owner_node = 'fake_owner_node'
    mock_get_state.return_value = dict(owner_node=owner_node)
    self.assertEqual(owner_node, self._clusterutils.get_vm_host(mock.sentinel.vm_name))
    self._cmgr.open_cluster_group.assert_called_once_with(mock.sentinel.vm_name)
    mock_get_state.assert_called_once_with(self._cmgr_val(self._cmgr.open_cluster_group))