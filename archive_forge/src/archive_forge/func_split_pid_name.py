from __future__ import (absolute_import, division, print_function)
import re
import platform
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import AnsibleModule
def split_pid_name(pid_name):
    """
    Split the entry PID/Program name into the PID (int) and the name (str)
    :param pid_name:  PID/Program String separated with a dash. E.g 51/sshd: returns pid = 51 and name = sshd
    :return: PID (int) and the program name (str)
    """
    try:
        pid, name = pid_name.split('/', 1)
    except ValueError:
        return (0, '')
    else:
        name = name.rstrip(':')
        return (int(pid), name)