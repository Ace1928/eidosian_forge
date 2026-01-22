from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from functools import partial
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def validate_roles(value, module):
    valid_roles = BUILTIN_ROLES + get_custom_roles(module)
    for item in value:
        if item not in valid_roles:
            module.fail_json(msg='invalid role specified')