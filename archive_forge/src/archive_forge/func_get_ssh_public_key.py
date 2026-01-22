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
def get_ssh_public_key(self):
    ssh_public_key_file = '%s.pub' % self.get_ssh_key_path()
    try:
        with open(ssh_public_key_file, 'r') as f:
            ssh_public_key = f.read().strip()
    except IOError:
        return None
    return ssh_public_key