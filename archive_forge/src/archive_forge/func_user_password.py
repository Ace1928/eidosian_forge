from __future__ import absolute_import, division, print_function
import ctypes.util
import grp
import calendar
import os
import re
import pty
import pwd
import select
import shutil
import socket
import subprocess
import time
import math
from ansible.module_utils import distro
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.sys_info import get_platform_subclass
import ansible.module_utils.compat.typing as t
def user_password(self):
    passwd = ''
    expires = ''
    if HAVE_SPWD:
        try:
            shadow_info = getspnam(to_bytes(self.name))
            passwd = to_native(shadow_info.sp_pwdp)
            expires = shadow_info.sp_expire
            return (passwd, expires)
        except ValueError:
            return (passwd, expires)
    if not self.user_exists():
        return (passwd, expires)
    elif self.SHADOWFILE:
        passwd, expires = self.parse_shadow_file()
    return (passwd, expires)