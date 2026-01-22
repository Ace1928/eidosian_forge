from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def convert_ip_to_network(self):
    """convert ip to subnet address"""
    ip_list = self.addr.split('.')
    mask_list = self.get_wildcard_mask().split('.')
    for i in range(len(ip_list)):
        ip_list[i] = str(int(ip_list[i]) & ~int(mask_list[i]) & 255)
    self.addr = '.'.join(ip_list)