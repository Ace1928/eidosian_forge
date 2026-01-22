from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def state_absent(portchannel, existing):
    commands = []
    if existing.get('vpc'):
        command = 'no vpc'
        commands.append(command)
    elif existing.get('peer-link'):
        command = 'no vpc peer-link'
        commands.append(command)
    if commands:
        commands.insert(0, 'interface port-channel{0}'.format(portchannel))
    return commands