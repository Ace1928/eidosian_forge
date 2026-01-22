from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
@mock.patch.object(vmutils.VMUtils, '_get_mounted_disk_resource_from_path')
def test_is_disk_attached(self, mock_get_mounted_disk_from_path):
    is_physical = True
    is_attached = self._vmutils.is_disk_attached(mock.sentinel.disk_path, is_physical=is_physical)
    self.assertTrue(is_attached)
    mock_get_mounted_disk_from_path.assert_called_once_with(mock.sentinel.disk_path, is_physical)