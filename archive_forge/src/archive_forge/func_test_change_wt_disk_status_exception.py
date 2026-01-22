from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.target import iscsi_target_utils as tg_utils
@mock.patch.object(tg_utils.ISCSITargetUtils, '_get_wt_disk')
def test_change_wt_disk_status_exception(self, mock_get_wt_disk):
    mock_wt_disk = mock_get_wt_disk.return_value
    mock_wt_disk.put.side_effect = test_base.FakeWMIExc
    wt_disk_enabled = True
    self.assertRaises(exceptions.ISCSITargetException, self._tgutils.change_wt_disk_status, mock.sentinel.wtd_name, enabled=wt_disk_enabled)
    mock_get_wt_disk.assert_called_once_with(mock.sentinel.wtd_name)
    self.assertEqual(wt_disk_enabled, mock_wt_disk.Enabled)