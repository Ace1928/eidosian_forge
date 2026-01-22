from __future__ import absolute_import, division, print_function
import platform
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.frr.frr.plugins.module_utils.network.frr.frr import (
def parse_interfaces(self, data):
    parsed = dict()
    key = ''
    for line in data.split('\n'):
        if len(line) == 0:
            continue
        elif line[0] == ' ':
            parsed[key] += '\n%s' % line
        else:
            match = re.match('^Interface (\\S+)', line)
            if match:
                key = match.group(1)
                parsed[key] = line
    return parsed