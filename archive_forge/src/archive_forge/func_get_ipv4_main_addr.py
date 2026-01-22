from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def get_ipv4_main_addr(self):
    """get IPv4 main address"""
    addrs = self.intf_info['am4CfgAddr']
    if not addrs:
        return None
    for address in addrs:
        if address['addrType'] == 'main':
            return address
    return None