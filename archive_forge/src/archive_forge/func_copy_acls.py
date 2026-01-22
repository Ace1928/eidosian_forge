import contextlib
import ctypes
import os
import shutil
import tempfile
from oslo_log import log as logging
from oslo_utils import fileutils
from os_win._i18n import _
from os_win import _utils
import os_win.conf
from os_win import exceptions
from os_win.utils import _acl_utils
from os_win.utils.io import ioutils
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi.libs import advapi32 as advapi32_def
from os_win.utils.winapi.libs import kernel32 as kernel32_def
from os_win.utils.winapi import wintypes
def copy_acls(self, source_path, dest_path):
    p_to_free = []
    try:
        sec_info_flags = w_const.DACL_SECURITY_INFORMATION
        sec_info = self._acl_utils.get_named_security_info(obj_name=source_path, obj_type=w_const.SE_FILE_OBJECT, security_info_flags=sec_info_flags)
        p_to_free.append(sec_info['pp_sec_desc'].contents)
        self._acl_utils.set_named_security_info(obj_name=dest_path, obj_type=w_const.SE_FILE_OBJECT, security_info_flags=sec_info_flags, p_dacl=sec_info['pp_dacl'].contents)
    finally:
        for p in p_to_free:
            self._win32_utils.local_free(p)