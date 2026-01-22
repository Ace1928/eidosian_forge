from __future__ import (absolute_import, division, print_function)
import re
from ansible_collections.community.routeros.plugins.module_utils.routeros import run_commands
from ansible_collections.community.routeros.plugins.module_utils.routeros import routeros_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
def parse_routing_mark(self, data):
    match = re.search('routing-mark=([\\w\\d\\-]+)', data, re.M)
    if match:
        return match.group(1)
    else:
        match = 'main'
        return match