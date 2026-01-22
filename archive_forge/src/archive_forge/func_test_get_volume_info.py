from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import hostutils
def test_get_volume_info(self):
    disk = mock.MagicMock()
    type(disk).Size = mock.PropertyMock(return_value=self._FAKE_DISK_SIZE)
    type(disk).FreeSpace = mock.PropertyMock(return_value=self._FAKE_DISK_FREE)
    self._hostutils._conn_cimv2.query.return_value = [disk]
    total_memory, free_memory = self._hostutils.get_volume_info(mock.sentinel.FAKE_DRIVE)
    self.assertEqual(self._FAKE_DISK_SIZE, total_memory)
    self.assertEqual(self._FAKE_DISK_FREE, free_memory)