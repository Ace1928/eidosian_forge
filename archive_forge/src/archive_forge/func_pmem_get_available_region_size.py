from __future__ import (absolute_import, division, print_function)
import json
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, human_to_bytes
def pmem_get_available_region_size(self, region):
    available_size = []
    for rg in region:
        available_size.append(rg['available_size'])
    return available_size