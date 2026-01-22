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
@ddt.data({}, {'expected_state': constants.CLUSTER_GROUP_OFFLINE, 'is_valid': False}, {'expected_node': 'some_other_node', 'is_valid': False})
@ddt.unpack
def test_validate_migration(self, expected_node=_FAKE_HOST, expected_state=constants.CLUSTER_GROUP_ONLINE, is_valid=True):
    group_state = dict(owner_node=self._FAKE_HOST.upper(), state=constants.CLUSTER_GROUP_ONLINE)
    self._clusapi.get_cluster_group_state.return_value = group_state
    if is_valid:
        self._clusterutils._validate_migration(mock.sentinel.group_handle, self._FAKE_VM_NAME, expected_state, expected_node)
    else:
        self.assertRaises(exceptions.ClusterGroupMigrationFailed, self._clusterutils._validate_migration, mock.sentinel.group_handle, self._FAKE_VM_NAME, expected_state, expected_node)
    self._clusapi.get_cluster_group_state.assert_called_once_with(mock.sentinel.group_handle)