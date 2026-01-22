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
@mock.patch.object(ctypes, 'byref')
def test_remove_persistent_login(self, mock_byref):
    fake_persistent_login = mock.Mock()
    fake_persistent_login.InitiatorInstance = 'fake_initiator_instance'
    fake_persistent_login.TargetName = 'fake_target_name'
    self._initiator._remove_persistent_login(fake_persistent_login)
    args_list = self._mock_run.call_args_list[0][0]
    self.assertIsInstance(args_list[1], ctypes.c_wchar_p)
    self.assertEqual(fake_persistent_login.InitiatorInstance, args_list[1].value)
    self.assertIsInstance(args_list[3], ctypes.c_wchar_p)
    self.assertEqual(fake_persistent_login.TargetName, args_list[3].value)
    mock_byref.assert_called_once_with(fake_persistent_login.TargetPortal)