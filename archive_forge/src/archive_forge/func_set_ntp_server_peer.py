from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def set_ntp_server_peer(peer_type, address, prefer, key_id, vrf_name):
    command_strings = []
    if prefer:
        command_strings.append(' prefer')
    if key_id:
        command_strings.append(' key {0}'.format(key_id))
    if vrf_name:
        command_strings.append(' use-vrf {0}'.format(vrf_name))
    command_strings.insert(0, 'ntp {0} {1}'.format(peer_type, address))
    command = ''.join(command_strings)
    return command