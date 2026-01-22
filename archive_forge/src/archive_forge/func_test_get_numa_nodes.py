from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import hostutils
@mock.patch.object(hostutils.HostUtils, '_get_numa_cpu_info')
@mock.patch.object(hostutils.HostUtils, '_get_numa_memory_info')
def test_get_numa_nodes(self, mock_get_memory_info, mock_get_cpu_info):
    numa_memory = mock_get_memory_info.return_value
    host_cpu = mock.MagicMock(DeviceID=self._DEVICE_ID)
    mock_get_cpu_info.return_value = [host_cpu]
    numa_node = mock.MagicMock(NodeID=self._NODE_ID)
    self._hostutils._conn.Msvm_NumaNode.return_value = [numa_node, numa_node]
    nodes_info = self._hostutils.get_numa_nodes()
    expected_info = {'id': self._DEVICE_ID.split('\\')[-1], 'memory': numa_memory.NumberOfBlocks, 'memory_usage': numa_node.CurrentlyConsumableMemoryBlocks, 'cpuset': set([self._DEVICE_ID.split('\\')[-1]]), 'cpu_usage': 0}
    self.assertEqual([expected_info, expected_info], nodes_info)