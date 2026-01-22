from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.target import iscsi_target_utils as tg_utils
@mock.patch.object(tg_utils.ISCSITargetUtils, '_get_wt_idmethod')
def test_deassociate_initiator_exception(self, mock_get_wtidmethod):
    mock_wt_idmetod = mock_get_wtidmethod.return_value
    mock_wt_idmetod.Delete_.side_effect = test_base.FakeWMIExc
    self.assertRaises(exceptions.ISCSITargetException, self._tgutils.deassociate_initiator, mock.sentinel.initiator, mock.sentinel.target_name)
    mock_get_wtidmethod.assert_called_once_with(mock.sentinel.initiator, mock.sentinel.target_name)