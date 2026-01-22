from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def parse_vgs(data):
    vgs = []
    for line in data.splitlines():
        parts = line.strip().split(';')
        vgs.append({'name': parts[0], 'size': float(parts[1]), 'free': float(parts[2]), 'ext_size': float(parts[3])})
    return vgs