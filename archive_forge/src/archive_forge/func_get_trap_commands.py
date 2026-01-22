from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_trap_commands(group, state, existing, module):
    commands = []
    enabled = False
    disabled = False
    if group == 'all':
        if state == 'disabled':
            for feature in existing:
                if existing[feature]:
                    trap_command = 'no snmp-server enable traps {0}'.format(feature)
                    commands.append(trap_command)
        elif state == 'enabled':
            for feature in existing:
                if existing[feature] is False:
                    trap_command = 'snmp-server enable traps {0}'.format(feature)
                    commands.append(trap_command)
    elif group in existing:
        if existing[group]:
            enabled = True
        else:
            disabled = True
        if state == 'disabled' and enabled:
            commands.append(['no snmp-server enable traps {0}'.format(group)])
        elif state == 'enabled' and disabled:
            commands.append(['snmp-server enable traps {0}'.format(group)])
    else:
        module.fail_json(msg='{0} is not a currently enabled feature.'.format(group))
    return commands