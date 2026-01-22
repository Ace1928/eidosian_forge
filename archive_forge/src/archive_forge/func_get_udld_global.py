from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_udld_global(module):
    command = 'show udld global | json'
    udld_table = run_commands(module, [command])[0]
    status = str(udld_table.get('udld-global-mode', None))
    if status == 'enabled-aggressive':
        aggressive = 'enabled'
    else:
        aggressive = 'disabled'
    interval = str(udld_table.get('message-interval', None))
    udld = dict(msg_time=interval, aggressive=aggressive)
    return udld