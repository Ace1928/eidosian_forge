from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage import diskutils
@mock.patch.object(diskutils.DiskUtils, '_get_disks_by_unique_id')
def test_get_disk_number_by_unique_id(self, mock_get_disks):
    mock_disks = [mock.Mock(), mock.Mock()]
    mock_get_disks.return_value = mock_disks
    exp_disk_numbers = [mock_disk.Number for mock_disk in mock_disks]
    returned_disk_numbers = self._diskutils.get_disk_numbers_by_unique_id(mock.sentinel.unique_id, mock.sentinel.unique_id_format)
    self.assertEqual(exp_disk_numbers, returned_disk_numbers)
    mock_get_disks.assert_called_once_with(mock.sentinel.unique_id, mock.sentinel.unique_id_format)