from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.ios import (
def parse_mode(module, config, group, member):
    mode = None
    netcfg = CustomNetworkConfig(indent=1, contents=config)
    parents = ['interface {0}'.format(member)]
    body = netcfg.get_section(parents)
    match_int = re.findall('interface {0}\\n'.format(member), body, re.M)
    if match_int:
        match = re.search('channel-group {0} mode (\\S+)'.format(group), body, re.M)
        if match:
            mode = match.group(1)
    return mode