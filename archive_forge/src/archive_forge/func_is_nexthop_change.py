from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def is_nexthop_change(self):
    """is ospf nexthop change"""
    if not self.ospf_info:
        return True
    for nexthop in self.ospf_info['nexthops']:
        if nexthop['ipAddress'] == self.nexthop_addr:
            if nexthop['weight'] == self.nexthop_weight:
                return False
            else:
                return True
    return True