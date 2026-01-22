from unittest import mock
import ddt
from os_win.tests.unit import test_base
from os_win.utils import _acl_utils
from os_win.utils.winapi import constants as w_const
@ddt.data({'security_info_flags': w_const.OWNER_SECURITY_INFORMATION | w_const.GROUP_SECURITY_INFORMATION | w_const.DACL_SECURITY_INFORMATION, 'expected_info': ['pp_sid_owner', 'pp_sid_group', 'pp_dacl', 'pp_sec_desc']}, {'security_info_flags': w_const.SACL_SECURITY_INFORMATION, 'expected_info': ['pp_sacl', 'pp_sec_desc']})
@ddt.unpack
@mock.patch.object(_acl_utils.ACLUtils, '_get_void_pp')
def test_get_named_security_info(self, mock_get_void_pp, security_info_flags, expected_info):
    sec_info = self._acl_utils.get_named_security_info(mock.sentinel.obj_name, mock.sentinel.obj_type, security_info_flags)
    self.assertEqual(set(expected_info), set(sec_info.keys()))
    for field in expected_info:
        self.assertEqual(sec_info[field], mock_get_void_pp.return_value)
    self._mock_run.assert_called_once_with(_acl_utils.advapi32.GetNamedSecurityInfoW, self._ctypes.c_wchar_p(mock.sentinel.obj_name), mock.sentinel.obj_type, security_info_flags, sec_info.get('pp_sid_owner'), sec_info.get('pp_sid_group'), sec_info.get('pp_dacl'), sec_info.get('pp_sacl'), sec_info['pp_sec_desc'])