from __future__ import absolute_import, division, print_function
import re
from ansible_collections.community.network.plugins.module_utils.network.cnos.cnos import run_commands
from ansible_collections.community.network.plugins.module_utils.network.cnos.cnos import check_args
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible.module_utils.six.moves import zip
def parse_ipaddresses(self, data):
    parsed = list()
    for line in data.split('\n'):
        if len(line) == 0:
            continue
        else:
            line = line.strip()
            match = re.match('^(Ethernet+)', line)
            if match:
                key = match.group(1)
                parsed.append(line)
            match = re.match('^(po+)', line)
            if match:
                key = match.group(1)
                parsed.append(line)
            match = re.match('^(mgmt+)', line)
            if match:
                key = match.group(1)
                parsed.append(line)
            match = re.match('^(loopback+)', line)
            if match:
                key = match.group(1)
                parsed.append(line)
    return parsed