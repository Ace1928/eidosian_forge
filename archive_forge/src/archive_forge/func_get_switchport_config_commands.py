from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import remove_default_spec
from ansible_collections.community.network.plugins.module_utils.network.slxos.slxos import get_config, load_config, run_commands
def get_switchport_config_commands(name, existing, proposed, module):
    """Gets commands required to config a given switchport interface
    """
    proposed_mode = proposed.get('mode')
    existing_mode = existing.get('mode')
    commands = []
    command = None
    if proposed_mode != existing_mode:
        if proposed_mode == 'trunk':
            command = 'switchport mode trunk'
        elif proposed_mode == 'access':
            command = 'switchport mode access'
    if command:
        commands.append(command)
    if proposed_mode == 'access':
        av_check = str(existing.get('access_vlan')) == str(proposed.get('access_vlan'))
        if not av_check:
            command = 'switchport access vlan {0}'.format(proposed.get('access_vlan'))
            commands.append(command)
    elif proposed_mode == 'trunk':
        tv_check = existing.get('trunk_vlans_list') == proposed.get('trunk_vlans_list')
        if not tv_check:
            if proposed.get('allowed'):
                command = 'switchport trunk allowed vlan add {0}'.format(proposed.get('trunk_allowed_vlans'))
                commands.append(command)
            else:
                existing_vlans = existing.get('trunk_vlans_list')
                proposed_vlans = proposed.get('trunk_vlans_list')
                vlans_to_add = set(proposed_vlans).difference(existing_vlans)
                if vlans_to_add:
                    command = 'switchport trunk allowed vlan add {0}'.format(proposed.get('trunk_vlans'))
                    commands.append(command)
        native_check = str(existing.get('native_vlan')) == str(proposed.get('native_vlan'))
        if not native_check and proposed.get('native_vlan'):
            command = 'switchport trunk native vlan {0}'.format(proposed.get('native_vlan'))
            commands.append(command)
    if commands:
        commands.insert(0, 'interface ' + name)
    return commands