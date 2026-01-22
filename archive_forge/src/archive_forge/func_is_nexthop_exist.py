from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def is_nexthop_exist(self):
    """is ospf nexthop exist"""
    if not self.ospf_info:
        return False
    for nexthop in self.ospf_info['nexthops']:
        if nexthop['ipAddress'] == self.nexthop_addr:
            return True
    return False