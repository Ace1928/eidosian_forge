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
def get_ssh_key_path(self):
    info = self.user_info()
    if os.path.isabs(self.ssh_file):
        ssh_key_file = self.ssh_file
    else:
        if not os.path.exists(info[5]) and (not self.module.check_mode):
            raise Exception('User %s home directory does not exist' % self.name)
        ssh_key_file = os.path.join(info[5], self.ssh_file)
    return ssh_key_file