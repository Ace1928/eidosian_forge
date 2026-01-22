from __future__ import (absolute_import, division, print_function)
import json
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, human_to_bytes
def get_allocation_result(self, goal, skt=None):
    ret = {'appdirect': 0, 'memorymode': 0}
    if skt:
        ret['socket'] = skt['id']
    out = xmltodict.parse(goal, dict_constructor=dict)['ConfigGoalList']['ConfigGoal']
    for entry in out:
        if skt and skt['id'] != int(entry['SocketID'], 16):
            continue
        for key, v in entry.items():
            if key == 'MemorySize':
                ret['memorymode'] += int(v.split()[0])
            elif key == 'AppDirect1Size' or key == 'AapDirect2Size':
                ret['appdirect'] += int(v.split()[0])
    capacity = self.pmem_get_capacity(skt)
    ret['reserved'] = capacity - ret['appdirect'] - ret['memorymode']
    return ret