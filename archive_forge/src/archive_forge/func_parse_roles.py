from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from functools import partial
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def parse_roles(data):
    configured_roles = None
    if 'TABLE_role' in data:
        configured_roles = data.get('TABLE_role')['ROW_role']
    roles = list()
    if configured_roles:
        for item in to_list(configured_roles):
            roles.append(item['role'])
    return roles