from __future__ import absolute_import, division, print_function
import re
import time
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import remove_default_spec
from ansible_collections.community.network.plugins.module_utils.network.cnos.cnos import load_config, run_commands
from ansible_collections.community.network.plugins.module_utils.network.cnos.cnos import debugOutput, check_args
from ansible_collections.community.network.plugins.module_utils.network.cnos.cnos import cnos_argument_spec
from ansible.module_utils._text import to_text
def parse_to_obj(logical_rows):
    first_row = logical_rows[0]
    rest_rows = logical_rows[1:]
    vlan_data = first_row.split()
    obj = {}
    obj['vlan_id'] = vlan_data[0]
    obj['name'] = vlan_data[1]
    obj['state'] = vlan_data[2]
    obj['interfaces'] = rest_rows
    return obj