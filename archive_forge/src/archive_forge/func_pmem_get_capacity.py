from __future__ import (absolute_import, division, print_function)
import json
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, human_to_bytes
def pmem_get_capacity(self, skt=None):
    command = ['show', '-d', 'Capacity', '-u', 'B', '-o', 'nvmxml', '-dimm']
    if skt:
        command += ['-socket', skt['id']]
    out = self.pmem_run_ipmctl(command)
    dimm_list = xmltodict.parse(out, dict_constructor=dict)['DimmList']['Dimm']
    capacity = 0
    for entry in dimm_list:
        for key, v in entry.items():
            if key == 'Capacity':
                capacity += int(v.split()[0])
    return capacity