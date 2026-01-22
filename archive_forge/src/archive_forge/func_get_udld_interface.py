from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_udld_interface(module, interface):
    command = 'show run udld all | section ' + interface.title() + '$'
    interface_udld = {}
    mode = None
    mode_str = None
    try:
        body = run_commands(module, [{'command': command, 'output': 'text'}])[0]
        if 'aggressive' in body:
            mode = 'aggressive'
            mode_str = 'aggressive'
        elif 'no udld enable' in body:
            mode = 'disabled'
            mode_str = 'no udld enable'
        elif 'no udld disable' in body:
            mode = 'enabled'
            mode_str = 'no udld disable'
        elif 'udld disable' in body:
            mode = 'disabled'
            mode_str = 'udld disable'
        elif 'udld enable' in body:
            mode = 'enabled'
            mode_str = 'udld enable'
        interface_udld['mode'] = mode
    except (KeyError, AttributeError, IndexError):
        interface_udld = {}
    return (interface_udld, mode_str)