from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_commands_config_udld_interface2(delta, interface, module, existing):
    commands = []
    existing, mode_str = get_udld_interface(module, interface)
    mode = delta['mode']
    if mode == 'enabled':
        if mode_str == 'no udld enable':
            command = 'udld enable'
        else:
            command = 'no udld disable'
    elif mode_str == 'no udld disable':
        command = 'udld disable'
    else:
        command = 'no udld enable'
    if command:
        commands.append(command)
        commands.insert(0, 'interface {0}'.format(interface))
    return commands