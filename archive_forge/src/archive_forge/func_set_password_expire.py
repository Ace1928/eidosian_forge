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
def set_password_expire(self):
    min_needs_change = self.password_expire_min is not None
    max_needs_change = self.password_expire_max is not None
    warn_needs_change = self.password_expire_warn is not None
    if HAVE_SPWD:
        try:
            shadow_info = getspnam(to_bytes(self.name))
        except ValueError:
            return (None, '', '')
        min_needs_change &= self.password_expire_min != shadow_info.sp_min
        max_needs_change &= self.password_expire_max != shadow_info.sp_max
        warn_needs_change &= self.password_expire_warn != shadow_info.sp_warn
    if not (min_needs_change or max_needs_change or warn_needs_change):
        return (None, '', '')
    command_name = 'chage'
    cmd = [self.module.get_bin_path(command_name, True)]
    if min_needs_change:
        cmd.extend(['-m', self.password_expire_min])
    if max_needs_change:
        cmd.extend(['-M', self.password_expire_max])
    if warn_needs_change:
        cmd.extend(['-W', self.password_expire_warn])
    cmd.append(self.name)
    return self.execute_command(cmd)