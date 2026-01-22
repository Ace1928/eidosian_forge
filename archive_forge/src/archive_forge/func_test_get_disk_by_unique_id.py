from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage import diskutils
def test_get_disk_by_unique_id(self):
    disk_cls = self._diskutils._conn_storage.Msft_Disk
    mock_disks = disk_cls.return_value
    resulted_disks = self._diskutils._get_disks_by_unique_id(mock.sentinel.unique_id, mock.sentinel.unique_id_format)
    disk_cls.assert_called_once_with(UniqueId=mock.sentinel.unique_id, UniqueIdFormat=mock.sentinel.unique_id_format)
    self.assertEqual(mock_disks, resulted_disks)