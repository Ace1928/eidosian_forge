from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
@ddt.data(None, mock.sentinel.serial)
def test_get_mounted_disk_resource_from_path(self, serial):
    mock_disk = mock.MagicMock()
    if serial:
        self._vmutils._conn.query.return_value = [mock_disk]
    else:
        mock_disk.HostResource = [self._FAKE_MOUNTED_DISK_PATH]
        self._vmutils._conn.query.return_value = [mock.MagicMock(), mock_disk]
    physical_disk = self._vmutils._get_mounted_disk_resource_from_path(self._FAKE_MOUNTED_DISK_PATH, True, serial=serial)
    self.assertEqual(mock_disk, physical_disk)