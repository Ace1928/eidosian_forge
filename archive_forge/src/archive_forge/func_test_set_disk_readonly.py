from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage import diskutils
@mock.patch.object(diskutils.DiskUtils, '_get_disk_by_number')
@ddt.data(0, 1)
def test_set_disk_readonly(self, err_code, mock_get_disk):
    mock_disk = mock_get_disk.return_value
    mock_disk.SetAttributes.return_value = (mock.sentinel.ext_err_info, err_code)
    if err_code:
        self.assertRaises(exceptions.DiskUpdateError, self._diskutils.set_disk_readonly_status, mock.sentinel.disk_number, read_only=True)
    else:
        self._diskutils.set_disk_readonly_status(mock.sentinel.disk_number, read_only=True)
    mock_disk.SetAttributes.assert_called_once_with(IsReadOnly=True)
    mock_get_disk.assert_called_once_with(mock.sentinel.disk_number)