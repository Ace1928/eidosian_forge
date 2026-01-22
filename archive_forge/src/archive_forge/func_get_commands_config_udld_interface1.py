from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_commands_config_udld_interface1(delta, interface, module, existing):
    commands = []
    mode = delta['mode']
    if mode == 'aggressive':
        commands.append('udld aggressive')
    else:
        commands.append('no udld aggressive')
    commands.insert(0, 'interface {0}'.format(interface))
    return commands