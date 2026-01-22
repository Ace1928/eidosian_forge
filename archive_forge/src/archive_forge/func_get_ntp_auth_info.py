from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_ntp_auth_info(key_id, module):
    auth_info = get_ntp_auth_key(key_id, module)
    trusted_key_list = get_ntp_trusted_key(module)
    auth_power = get_ntp_auth(module)
    if key_id in trusted_key_list:
        auth_info['trusted_key'] = 'true'
    else:
        auth_info['trusted_key'] = 'false'
    if auth_power:
        auth_info['authentication'] = 'on'
    else:
        auth_info['authentication'] = 'off'
    return auth_info