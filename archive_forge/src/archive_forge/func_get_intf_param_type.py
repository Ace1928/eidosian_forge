from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, get_nc_config, get_nc_next
def get_intf_param_type(self):
    """Get the type of input interface parameter"""
    if self.interface == 'all':
        self.param_type = INTERFACE_ALL
        return
    if self.if_type == self.interface:
        self.param_type = INTERFACE_TYPE
        return
    self.param_type = INTERFACE_FULL_NAME