from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_commands_remove_udld_global(existing):
    commands = []
    if existing.get('aggressive') == 'enabled':
        command = 'no udld aggressive'
        commands.append(command)
    if existing.get('msg_time') != PARAM_TO_DEFAULT_KEYMAP.get('msg_time'):
        command = 'no udld message-time'
        commands.append(command)
    return commands