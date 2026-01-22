from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def fix_proposed(proposed_commands):
    new_proposed = {}
    for key, value in proposed_commands.items():
        if key == 'route-target both':
            new_proposed['route-target export'] = value
            new_proposed['route-target import'] = value
        else:
            new_proposed[key] = value
    return new_proposed