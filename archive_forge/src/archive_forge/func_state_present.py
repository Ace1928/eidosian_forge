from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def state_present(portchannel, delta, config_value, existing):
    commands = []
    command = get_commands_to_config_vpc_interface(portchannel, delta, config_value, existing)
    commands.append(command)
    return commands