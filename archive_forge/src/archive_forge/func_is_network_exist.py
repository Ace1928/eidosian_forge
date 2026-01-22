from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def is_network_exist(self):
    """is ospf area network exist"""
    if not self.ospf_info:
        return False
    for area in self.ospf_info['areas']:
        if area['areaId'] == self.get_area_ip():
            if not area.get('networks'):
                return False
            for network in area.get('networks'):
                if network['ipAddress'] == self.addr and network['wildcardMask'] == self.get_wildcard_mask():
                    return True
    return False