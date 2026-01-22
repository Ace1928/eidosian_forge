import ctypes
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
def get_named_security_info(self, obj_name, obj_type, security_info_flags):
    """Retrieve object security information.

        :param security_info_flags: specifies which information will
                                   be retrieved.
        :param ret_val: dict, containing pointers to the requested structures.
                        Note that the returned security descriptor will have
                        to be freed using LocalFree.
                        Some requested information may not be present, in
                        which case the according pointers will be NULL.
        """
    sec_info = {}
    if security_info_flags & w_const.OWNER_SECURITY_INFORMATION:
        sec_info['pp_sid_owner'] = self._get_void_pp()
    if security_info_flags & w_const.GROUP_SECURITY_INFORMATION:
        sec_info['pp_sid_group'] = self._get_void_pp()
    if security_info_flags & w_const.DACL_SECURITY_INFORMATION:
        sec_info['pp_dacl'] = self._get_void_pp()
    if security_info_flags & w_const.SACL_SECURITY_INFORMATION:
        sec_info['pp_sacl'] = self._get_void_pp()
    sec_info['pp_sec_desc'] = self._get_void_pp()
    self._win32_utils.run_and_check_output(advapi32.GetNamedSecurityInfoW, ctypes.c_wchar_p(obj_name), obj_type, security_info_flags, sec_info.get('pp_sid_owner'), sec_info.get('pp_sid_group'), sec_info.get('pp_dacl'), sec_info.get('pp_sacl'), sec_info['pp_sec_desc'])
    return sec_info