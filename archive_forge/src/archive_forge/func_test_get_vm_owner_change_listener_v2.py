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
@mock.patch.object(clusterutils, '_ClusterGroupOwnerChangeListener')
@mock.patch.object(clusterutils.ClusterUtils, 'get_cluster_node_name')
@mock.patch.object(clusterutils.ClusterUtils, 'get_cluster_group_type')
@mock.patch.object(clusterutils, 'time')
def test_get_vm_owner_change_listener_v2(self, mock_time, mock_get_type, mock_get_node_name, mock_listener):
    mock_get_type.side_effect = [w_const.ClusGroupTypeVirtualMachine, mock.sentinel.other_type]
    mock_events = [mock.MagicMock(), mock.MagicMock()]
    mock_listener.return_value.get.side_effect = mock_events + [exceptions.OSWinException, KeyboardInterrupt]
    callback = mock.Mock()
    listener = self._clusterutils.get_vm_owner_change_listener_v2()
    self.assertRaises(KeyboardInterrupt, listener, callback)
    callback.assert_called_once_with(mock_events[0]['cluster_object_name'], mock_get_node_name.return_value)
    mock_listener.assert_called_once_with(self._clusapi.open_cluster.return_value)
    mock_get_node_name.assert_called_once_with(mock_events[0]['parent_id'])
    mock_get_type.assert_any_call(mock_events[0]['cluster_object_name'])
    mock_time.sleep.assert_called_once_with(constants.DEFAULT_WMI_EVENT_TIMEOUT_MS / 1000)