from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.target import iscsi_target_utils as tg_utils
@mock.patch.object(tg_utils.ISCSITargetUtils, '_get_wt_disk')
def test_create_snapshot_exception(self, mock_get_wt_disk):
    mock_wt_disk = mock_get_wt_disk.return_value
    mock_wt_snap = mock.Mock()
    mock_wt_snap.put.side_effect = test_base.FakeWMIExc
    mock_wt_snap_cls = self._tgutils._conn_wmi.WT_Snapshot
    mock_wt_snap_cls.return_value = [mock_wt_snap]
    mock_wt_snap_cls.Create.return_value = [mock.sentinel.snap_id]
    self.assertRaises(exceptions.ISCSITargetException, self._tgutils.create_snapshot, mock.sentinel.wtd_name, mock.sentinel.snap_name)
    mock_get_wt_disk.assert_called_once_with(mock.sentinel.wtd_name)
    mock_wt_snap_cls.Create.assert_called_once_with(WTD=mock_wt_disk.WTD)
    mock_wt_snap_cls.assert_called_once_with(Id=mock.sentinel.snap_id)
    self.assertEqual(mock.sentinel.snap_name, mock_wt_snap.Description)