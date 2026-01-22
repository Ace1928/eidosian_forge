from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import hostutils
def test_get_numa_cpu_info(self):
    host_cpu = mock.MagicMock()
    host_cpu.path_.return_value = 'fake_wmi_obj_path'
    vm_cpu = mock.MagicMock()
    vm_cpu.path_.return_value = 'fake_wmi_obj_path1'
    numa_node_assoc = [host_cpu]
    cpu_info = self._hostutils._get_numa_cpu_info(numa_node_assoc, [host_cpu, vm_cpu])
    self.assertEqual([host_cpu], cpu_info)