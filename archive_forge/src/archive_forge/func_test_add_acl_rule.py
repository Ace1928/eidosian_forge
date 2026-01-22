import ctypes
import os
import shutil
from unittest import mock
import ddt
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import pathutils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi.libs import advapi32 as advapi32_def
from os_win.utils.winapi.libs import kernel32 as kernel32_def
from os_win.utils.winapi import wintypes
def test_add_acl_rule(self):
    raised_exc = exceptions.OSWinException
    self._ctypes_patcher.stop()
    fake_trustee = 'FAKEDOMAIN\\FakeUser'
    mock_sec_info = dict(pp_sec_desc=mock.Mock(), pp_dacl=mock.Mock())
    self._acl_utils.get_named_security_info.return_value = mock_sec_info
    self._acl_utils.set_named_security_info.side_effect = raised_exc
    pp_new_dacl = self._acl_utils.set_entries_in_acl.return_value
    self.assertRaises(raised_exc, self._pathutils.add_acl_rule, path=mock.sentinel.path, trustee_name=fake_trustee, access_rights=constants.ACE_GENERIC_READ, access_mode=constants.ACE_GRANT_ACCESS, inheritance_flags=constants.ACE_OBJECT_INHERIT)
    self._acl_utils.get_named_security_info.assert_called_once_with(obj_name=mock.sentinel.path, obj_type=w_const.SE_FILE_OBJECT, security_info_flags=w_const.DACL_SECURITY_INFORMATION)
    self._acl_utils.set_entries_in_acl.assert_called_once_with(entry_count=1, p_explicit_entry_list=mock.ANY, p_old_acl=mock_sec_info['pp_dacl'].contents)
    self._acl_utils.set_named_security_info.assert_called_once_with(obj_name=mock.sentinel.path, obj_type=w_const.SE_FILE_OBJECT, security_info_flags=w_const.DACL_SECURITY_INFORMATION, p_dacl=pp_new_dacl.contents)
    p_access = self._acl_utils.set_entries_in_acl.call_args_list[0][1]['p_explicit_entry_list']
    access = ctypes.cast(p_access, ctypes.POINTER(advapi32_def.EXPLICIT_ACCESS)).contents
    self.assertEqual(constants.ACE_GENERIC_READ, access.grfAccessPermissions)
    self.assertEqual(constants.ACE_GRANT_ACCESS, access.grfAccessMode)
    self.assertEqual(constants.ACE_OBJECT_INHERIT, access.grfInheritance)
    self.assertEqual(w_const.TRUSTEE_IS_NAME, access.Trustee.TrusteeForm)
    self.assertEqual(fake_trustee, access.Trustee.pstrName)
    self._pathutils._win32_utils.local_free.assert_has_calls([mock.call(pointer) for pointer in [mock_sec_info['pp_sec_desc'].contents, pp_new_dacl.contents]])