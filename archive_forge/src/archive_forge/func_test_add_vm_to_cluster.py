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
@ddt.data(True, False)
@mock.patch.object(clusterutils.ClusterUtils, '_lookup_vm_group_check')
def test_add_vm_to_cluster(self, auto_failback, mock_lookup_vm_group_check):
    self._clusterutils._cluster.AddVirtualMachine = mock.MagicMock()
    vm_group = mock.Mock()
    mock_lookup_vm_group_check.return_value = vm_group
    self._clusterutils.add_vm_to_cluster(self._FAKE_VM_NAME, mock.sentinel.max_failover_count, mock.sentinel.failover_period, auto_failback)
    self.assertEqual(mock.sentinel.max_failover_count, vm_group.FailoverThreshold)
    self.assertEqual(mock.sentinel.failover_period, vm_group.FailoverPeriod)
    self.assertTrue(vm_group.PersistentState)
    self.assertEqual(vm_group.AutoFailbackType, int(auto_failback))
    self.assertEqual(vm_group.FailbackWindowStart, self._clusterutils._FAILBACK_WINDOW_MIN)
    self.assertEqual(vm_group.FailbackWindowEnd, self._clusterutils._FAILBACK_WINDOW_MAX)
    vm_group.put.assert_called_once_with()