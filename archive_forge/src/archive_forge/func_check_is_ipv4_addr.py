from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import exec_command, load_config, ce_argument_spec
def check_is_ipv4_addr(self):
    """Check ipaddress validate"""
    rule1 = '(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])\\.'
    rule2 = '(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])'
    ipv4_regex = '%s%s%s%s%s%s' % ('^', rule1, rule1, rule1, rule2, '$')
    return bool(re.match(ipv4_regex, self.peer))