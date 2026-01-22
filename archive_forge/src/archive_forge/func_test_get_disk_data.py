import platform
from unittest import mock
import ddt
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import livemigrationutils
from os_win.utils.compute import vmutils
from os_win.utils import jobutils
def test_get_disk_data(self):
    mock_vmutils_remote = mock.MagicMock()
    mock_disk = mock.MagicMock()
    mock_disk_path_mapping = {mock.sentinel.serial: mock.sentinel.disk_path}
    mock_disk.path.return_value.RelPath = mock.sentinel.rel_path
    mock_vmutils_remote.get_vm_disks.return_value = [None, [mock_disk]]
    mock_disk.ElementName = mock.sentinel.serial
    resulted_disk_paths = self.liveutils._get_disk_data(self._FAKE_VM_NAME, mock_vmutils_remote, mock_disk_path_mapping)
    mock_vmutils_remote.get_vm_disks.assert_called_once_with(self._FAKE_VM_NAME)
    mock_disk.path.assert_called_once_with()
    expected_disk_paths = {mock.sentinel.rel_path: mock.sentinel.disk_path}
    self.assertEqual(expected_disk_paths, resulted_disk_paths)