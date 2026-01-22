from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def fix_delta(delta, existing):
    for key in list(delta):
        if key in ['dr_prio', 'hello_interval', 'sparse', 'border']:
            if delta.get(key) == PARAM_TO_DEFAULT_KEYMAP.get(key) and existing.get(key) is None:
                delta.pop(key)
    return delta