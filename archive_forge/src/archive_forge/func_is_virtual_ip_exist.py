from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def is_virtual_ip_exist(self):
    """whether virtual ip info exist"""
    if not self.virtual_ip_info:
        return False
    for info in self.virtual_ip_info['vrrpVirtualIpInfos']:
        if info['virtualIpAddress'] == self.virtual_ip:
            return True
    return False