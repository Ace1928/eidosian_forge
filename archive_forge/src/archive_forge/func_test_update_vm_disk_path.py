from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
@mock.patch.object(vmutils.VMUtils, '_get_mounted_disk_resource_from_path')
def test_update_vm_disk_path(self, mock_get_disk_resource_from_path):
    disk_resource = mock_get_disk_resource_from_path.return_value
    self._vmutils.update_vm_disk_path(mock.sentinel.disk_path, mock.sentinel.new_path, is_physical=True)
    mock_get_disk_resource_from_path.assert_called_once_with(disk_path=mock.sentinel.disk_path, is_physical=True)
    self._vmutils._jobutils.modify_virt_resource.assert_called_once_with(disk_resource)
    self.assertEqual(disk_resource.HostResource, [mock.sentinel.new_path])