from unittest import mock
import ddt
from os_win.tests.unit import test_base
from os_win.utils import _acl_utils
from os_win.utils.winapi import constants as w_const
def test_set_named_security_info(self):
    self._acl_utils.set_named_security_info(mock.sentinel.obj_name, mock.sentinel.obj_type, mock.sentinel.security_info_flags, mock.sentinel.p_sid_owner, mock.sentinel.p_sid_group, mock.sentinel.p_dacl, mock.sentinel.p_sacl)
    self._mock_run.assert_called_once_with(_acl_utils.advapi32.SetNamedSecurityInfoW, self._ctypes.c_wchar_p(mock.sentinel.obj_name), mock.sentinel.obj_type, mock.sentinel.security_info_flags, mock.sentinel.p_sid_owner, mock.sentinel.p_sid_group, mock.sentinel.p_dacl, mock.sentinel.p_sacl)