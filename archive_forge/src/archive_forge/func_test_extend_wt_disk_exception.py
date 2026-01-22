from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.target import iscsi_target_utils as tg_utils
@mock.patch.object(tg_utils.ISCSITargetUtils, '_get_wt_disk')
def test_extend_wt_disk_exception(self, mock_get_wt_disk):
    mock_wt_disk = mock_get_wt_disk.return_value
    mock_wt_disk.Extend.side_effect = test_base.FakeWMIExc
    self.assertRaises(exceptions.ISCSITargetException, self._tgutils.extend_wt_disk, mock.sentinel.wtd_name, mock.sentinel.additional_mb)
    mock_get_wt_disk.assert_called_once_with(mock.sentinel.wtd_name)
    mock_wt_disk.Extend.assert_called_once_with(mock.sentinel.additional_mb)