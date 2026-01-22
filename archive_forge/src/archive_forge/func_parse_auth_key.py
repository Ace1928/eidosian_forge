from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.ios import (
def parse_auth_key(line, dest):
    if dest == 'authentication-key':
        match = re.search('(ntp\\sauthentication-key\\s\\d+\\smd5\\s)(\\w+)', line, re.M)
        if match:
            auth_key = match.group(2)
            return auth_key