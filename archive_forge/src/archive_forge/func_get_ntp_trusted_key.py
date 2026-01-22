from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_ntp_trusted_key(module):
    trusted_key_list = []
    command = 'show run | inc ntp.trusted-key'
    trusted_key_str = execute_show_command(command, module)[0]
    if trusted_key_str:
        trusted_keys = trusted_key_str.splitlines()
    else:
        trusted_keys = []
    for line in trusted_keys:
        if line:
            trusted_key_list.append(str(line.split()[2]))
    return trusted_key_list