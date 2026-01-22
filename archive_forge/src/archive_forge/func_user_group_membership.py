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
def user_group_membership(self, exclude_primary=True):
    """ Return a list of groups the user belongs to """
    groups = []
    info = self.get_pwd_info()
    for group in grp.getgrall():
        if self.name in group.gr_mem:
            if not exclude_primary:
                groups.append(group[0])
            elif info[3] != group.gr_gid:
                groups.append(group[0])
    return groups