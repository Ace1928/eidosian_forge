from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.ios import (
def parse_domain_search(config):
    match = re.findall('^ip domain[- ]list (?:vrf (\\S+) )*(\\S+)', config, re.M)
    matches = list()
    for vrf, name in match:
        if not vrf:
            vrf = None
        matches.append({'name': name, 'vrf': vrf})
    return matches