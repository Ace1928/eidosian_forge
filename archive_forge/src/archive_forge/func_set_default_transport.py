from __future__ import (absolute_import, division, print_function)
import subprocess
from ansible import constants as C
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.compat.paramiko import paramiko
from ansible.utils.display import Display
def set_default_transport():
    if C.DEFAULT_TRANSPORT == 'smart':
        display.deprecated("The 'smart' option for connections is deprecated. Set the connection plugin directly instead.", version='2.20')
        if not check_for_controlpersist('ssh') and paramiko is not None:
            C.DEFAULT_TRANSPORT = 'paramiko'
        else:
            C.DEFAULT_TRANSPORT = 'ssh'