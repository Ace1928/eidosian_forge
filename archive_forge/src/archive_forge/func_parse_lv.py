from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def parse_lv(data):
    name = None
    for line in data.splitlines():
        match = re.search('LOGICAL VOLUME:\\s+(\\w+)\\s+VOLUME GROUP:\\s+(\\w+)', line)
        if match is not None:
            name = match.group(1)
            vg = match.group(2)
            continue
        match = re.search('LPs:\\s+(\\d+).*PPs', line)
        if match is not None:
            lps = int(match.group(1))
            continue
        match = re.search('PP SIZE:\\s+(\\d+)', line)
        if match is not None:
            pp_size = int(match.group(1))
            continue
        match = re.search('INTER-POLICY:\\s+(\\w+)', line)
        if match is not None:
            policy = match.group(1)
            continue
    if not name:
        return None
    size = lps * pp_size
    return {'name': name, 'vg': vg, 'size': size, 'policy': policy}