from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_interface_mode(interface, intf_type, module):
    mode = 'unknown'
    command = 'show interface {0}'.format(interface)
    body = execute_show_command(command, module)
    try:
        interface_table = body[0]['TABLE_interface']['ROW_interface']
    except (KeyError, AttributeError, IndexError):
        return mode
    if intf_type in ['ethernet', 'portchannel']:
        mode = str(interface_table.get('eth_mode', 'layer3'))
        if mode in ['access', 'trunk']:
            mode = 'layer2'
        elif mode == 'routed':
            mode = 'layer3'
    elif intf_type in ['loopback', 'svi']:
        mode = 'layer3'
    return mode