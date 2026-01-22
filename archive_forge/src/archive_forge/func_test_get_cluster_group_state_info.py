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
@mock.patch.object(clusterutils.ClusterUtils, '_is_migration_queued')
def test_get_cluster_group_state_info(self, mock_is_migr_queued, mock_get_gr_state):
    exp_clus_group_h = self._cmgr_val(self._cmgr.open_cluster_group)
    mock_get_gr_state.return_value = dict(state=mock.sentinel.state, status_info=mock.sentinel.status_info, owner_node=mock.sentinel.owner_node)
    sts_info = self._clusterutils.get_cluster_group_state_info(mock.sentinel.group_name)
    exp_sts_info = dict(state=mock.sentinel.state, owner_node=mock.sentinel.owner_node, migration_queued=mock_is_migr_queued.return_value)
    self.assertEqual(exp_sts_info, sts_info)
    self._cmgr.open_cluster_group.assert_called_once_with(mock.sentinel.group_name)
    mock_get_gr_state.assert_called_once_with(exp_clus_group_h)
    mock_is_migr_queued.assert_called_once_with(mock.sentinel.status_info)