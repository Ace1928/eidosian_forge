from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_group_timeout(config):
    match = re.search('  Group timeout configured: (\\S+)', config, re.M)
    if match:
        value = match.group(1)
    else:
        value = ''
    return value