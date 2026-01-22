from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import remove_default_spec
from ansible_collections.community.network.plugins.module_utils.network.slxos.slxos import get_config, load_config, run_commands
def remove_switchport_config_commands(name, existing, proposed, module):
    mode = proposed.get('mode')
    commands = []
    command = None
    if mode == 'access':
        av_check = existing.get('access_vlan') == proposed.get('access_vlan')
        if av_check:
            command = 'no switchport access vlan {0}'.format(existing.get('access_vlan'))
            commands.append(command)
    elif mode == 'trunk':
        tv_check = existing.get('trunk_vlans_list') == proposed.get('trunk_vlans_list')
        if not tv_check:
            existing_vlans = existing.get('trunk_vlans_list')
            proposed_vlans = proposed.get('trunk_vlans_list')
            vlans_to_remove = set(proposed_vlans).intersection(existing_vlans)
            if vlans_to_remove:
                proposed_allowed_vlans = proposed.get('trunk_allowed_vlans')
                remove_trunk_allowed_vlans = proposed.get('trunk_vlans', proposed_allowed_vlans)
                command = 'switchport trunk allowed vlan remove {0}'.format(remove_trunk_allowed_vlans)
                commands.append(command)
        native_check = existing.get('native_vlan') == proposed.get('native_vlan')
        if native_check and proposed.get('native_vlan'):
            command = 'no switchport trunk native vlan {0}'.format(existing.get('native_vlan'))
            commands.append(command)
    if commands:
        commands.insert(0, 'interface ' + name)
    return commands