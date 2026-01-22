from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage import diskutils
@mock.patch.object(diskutils.DiskUtils, '_get_disk_by_number')
def test_refresh_disk(self, mock_get_disk):
    mock_disk = mock_get_disk.return_value
    self._diskutils.refresh_disk(mock.sentinel.disk_number)
    mock_get_disk.assert_called_once_with(mock.sentinel.disk_number)
    mock_disk.Refresh.assert_called_once_with()