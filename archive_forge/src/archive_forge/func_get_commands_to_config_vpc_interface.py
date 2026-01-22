from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_commands_to_config_vpc_interface(portchannel, delta, config_value, existing):
    commands = []
    if not delta.get('peer-link') and existing.get('peer-link'):
        commands.append('no vpc peer-link')
        commands.insert(0, 'interface port-channel{0}'.format(portchannel))
    elif delta.get('peer-link') and (not existing.get('peer-link')):
        commands.append('vpc peer-link')
        commands.insert(0, 'interface port-channel{0}'.format(portchannel))
    elif delta.get('vpc') and (not existing.get('vpc')):
        command = 'vpc {0}'.format(config_value)
        commands.append(command)
        commands.insert(0, 'interface port-channel{0}'.format(portchannel))
    return commands