from __future__ import (absolute_import, division, print_function)
import json
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, human_to_bytes
def pmem_get_available_region_type(self, region):
    types = []
    for rg in region:
        if rg['type'] not in types:
            types.append(rg['type'])
    return types