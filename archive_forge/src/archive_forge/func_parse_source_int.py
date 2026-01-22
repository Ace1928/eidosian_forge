from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.ios import (
def parse_source_int(line, dest):
    if dest == 'source':
        match = re.search('(ntp\\ssource\\s)(\\S+)', line, re.M)
        if match:
            source = match.group(2)
            return source