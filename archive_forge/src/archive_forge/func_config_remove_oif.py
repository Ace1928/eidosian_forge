from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def config_remove_oif(existing, existing_oif_prefix_source):
    commands = []
    command = None
    if existing.get('oif_routemap'):
        commands.append('no ip igmp static-oif route-map {0}'.format(existing.get('oif_routemap')))
    elif existing_oif_prefix_source:
        for each in existing_oif_prefix_source:
            if each.get('prefix') and each.get('source'):
                command = 'no ip igmp static-oif {0} source {1} '.format(each.get('prefix'), each.get('source'))
            elif each.get('prefix'):
                command = 'no ip igmp static-oif {0}'.format(each.get('prefix'))
            if command:
                commands.append(command)
            command = None
    return commands