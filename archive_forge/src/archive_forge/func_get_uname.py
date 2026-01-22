from __future__ import (absolute_import, division, print_function)
import os
import platform
import re
import ansible.module_utils.compat.typing as t
from ansible.module_utils.common.sys_info import get_distribution, get_distribution_version, \
from ansible.module_utils.facts.utils import get_file_content, get_file_lines
from ansible.module_utils.facts.collector import BaseFactCollector
def get_uname(module, flags='-v'):
    if isinstance(flags, str):
        flags = flags.split()
    command = ['uname']
    command.extend(flags)
    rc, out, err = module.run_command(command)
    if rc == 0:
        return out
    return None