from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.target import iscsi_target_utils as tg_utils
@mock.patch.object(tg_utils.ISCSITargetUtils, '_get_wt_host')
def test_set_chap_credentials_exception(self, mock_get_wt_host):
    mock_wt_host = mock_get_wt_host.return_value
    mock_wt_host.put.side_effect = test_base.FakeWMIExc
    self.assertRaises(exceptions.ISCSITargetException, self._tgutils.set_chap_credentials, mock.sentinel.target_name, mock.sentinel.chap_username, mock.sentinel.chap_password)
    mock_get_wt_host.assert_called_once_with(mock.sentinel.target_name)
    (self.assertTrue(mock_wt_host.EnableCHAP),)
    self.assertEqual(mock.sentinel.chap_username, mock_wt_host.CHAPUserName)
    self.assertEqual(mock.sentinel.chap_password, mock_wt_host.CHAPSecret)
    mock_wt_host.put.assert_called_once_with()