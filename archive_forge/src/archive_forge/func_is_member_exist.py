from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def is_member_exist(self, ifname):
    """is trunk member exist"""
    if not self.trunk_info['TrunkMemberIfs']:
        return False
    for mem in self.trunk_info['TrunkMemberIfs']:
        if ifname.replace(' ', '').upper() == mem['memberIfName'].replace(' ', '').upper():
            return True
    return False