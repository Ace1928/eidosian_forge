from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def config_default_igmp_interface(existing, delta):
    commands = []
    proposed = get_igmp_interface_defaults()
    delta = dict(set(proposed.items()).difference(existing.items()))
    if delta:
        command = config_igmp_interface(delta, existing, existing_oif_prefix_source=None)
        if command:
            for each in command:
                commands.append(each)
    return commands