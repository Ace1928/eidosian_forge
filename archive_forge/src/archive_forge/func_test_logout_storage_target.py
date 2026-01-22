import collections
import ctypes
from unittest import mock
import ddt
import six
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.initiator import iscsi_utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi.errmsg import iscsierr
from os_win.utils.winapi.libs import iscsidsc as iscsi_struct
@mock.patch.object(iscsi_utils.ISCSIInitiatorUtils, '_get_iscsi_target_sessions')
@mock.patch.object(iscsi_utils.ISCSIInitiatorUtils, '_logout_iscsi_target')
@mock.patch.object(iscsi_utils.ISCSIInitiatorUtils, '_remove_target_persistent_logins')
@mock.patch.object(iscsi_utils.ISCSIInitiatorUtils, '_remove_static_target')
def test_logout_storage_target(self, mock_remove_static_target, mock_remove_target_persistent_logins, mock_logout_iscsi_target, mock_get_iscsi_target_sessions):
    fake_session = mock.Mock(SessionId=mock.sentinel.session_id)
    mock_get_iscsi_target_sessions.return_value = [fake_session]
    self._initiator.logout_storage_target(mock.sentinel.target_iqn)
    mock_get_iscsi_target_sessions.assert_called_once_with(mock.sentinel.target_iqn, connected_only=False)
    mock_logout_iscsi_target.assert_called_once_with(mock.sentinel.session_id)
    mock_remove_target_persistent_logins.assert_called_once_with(mock.sentinel.target_iqn)
    mock_remove_static_target.assert_called_once_with(mock.sentinel.target_iqn)