from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.target import iscsi_target_utils as tg_utils
@mock.patch.object(tg_utils.ISCSITargetUtils, '_get_wt_idmethod')
def test_associate_initiator_exception(self, mock_get_wtidmethod):
    mock_get_wtidmethod.return_value = None
    mock_wt_idmeth_cls = self._tgutils._conn_wmi.WT_IDMethod
    mock_wt_idmetod = mock_wt_idmeth_cls.new.return_value
    mock_wt_idmetod.put.side_effect = test_base.FakeWMIExc
    self.assertRaises(exceptions.ISCSITargetException, self._tgutils.associate_initiator_with_iscsi_target, mock.sentinel.initiator, mock.sentinel.target_name, id_method=mock.sentinel.id_method)
    self.assertEqual(mock.sentinel.target_name, mock_wt_idmetod.HostName)
    self.assertEqual(mock.sentinel.initiator, mock_wt_idmetod.Value)
    self.assertEqual(mock.sentinel.id_method, mock_wt_idmetod.Method)
    mock_get_wtidmethod.assert_called_once_with(mock.sentinel.initiator, mock.sentinel.target_name)