from __future__ import absolute_import, division, print_function
import re
import time
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def validate_vrf(name, module):
    if name:
        name = name.strip()
        if name == 'default':
            module.fail_json(msg='cannot use default as name of a VRF')
        elif len(name) > 32:
            module.fail_json(msg='VRF name exceeded max length of 32', name=name)
        else:
            return name