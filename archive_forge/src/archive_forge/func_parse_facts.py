from __future__ import absolute_import, division, print_function
import platform
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.frr.frr.plugins.module_utils.network.frr.frr import (
def parse_facts(self, pattern, data):
    value = None
    match = re.search(pattern, data, re.M)
    if match:
        value = match.group(1)
    return value