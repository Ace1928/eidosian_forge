from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage import diskutils
@ddt.data(True, False)
def test_get_disk_by_number(self, msft_disk_cls):
    resulted_disk = self._diskutils._get_disk_by_number(mock.sentinel.disk_number, msft_disk_cls=msft_disk_cls)
    if msft_disk_cls:
        disk_cls = self._diskutils._conn_storage.Msft_Disk
        disk_cls.assert_called_once_with(Number=mock.sentinel.disk_number)
    else:
        disk_cls = self._diskutils._conn_cimv2.Win32_DiskDrive
        disk_cls.assert_called_once_with(Index=mock.sentinel.disk_number)
    mock_disk = disk_cls.return_value[0]
    self.assertEqual(mock_disk, resulted_disk)