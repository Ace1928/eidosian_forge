from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.target import iscsi_target_utils as tg_utils
@mock.patch.object(tg_utils.ISCSITargetUtils, '_get_wt_host')
def test_get_target_information(self, mock_get_wt_host):
    mock_wt_host = mock_get_wt_host.return_value
    mock_wt_host.EnableCHAP = True
    mock_wt_host.Status = 1
    target_info = self._tgutils.get_target_information(mock.sentinel.target_name)
    expected_info = dict(target_iqn=mock_wt_host.TargetIQN, enabled=mock_wt_host.Enabled, connected=True, auth_method='CHAP', auth_username=mock_wt_host.CHAPUserName, auth_password=mock_wt_host.CHAPSecret)
    self.assertEqual(expected_info, target_info)
    mock_get_wt_host.assert_called_once_with(mock.sentinel.target_name)