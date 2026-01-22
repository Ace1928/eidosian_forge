from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def is_vrrp_global_info_change(self):
    """whether vrrp global attribute info change"""
    if not self.vrrp_global_info:
        return True
    if self.gratuitous_arp_interval:
        if self.vrrp_global_info['gratuitousArpFlag'] == 'false':
            self.module.fail_json(msg='Error: gratuitousArpFlag is false.')
        if self.vrrp_global_info['gratuitousArpTimeOut'] != self.gratuitous_arp_interval:
            return True
    if self.recover_delay:
        if self.vrrp_global_info['recoverDelay'] != self.recover_delay:
            return True
    if self.version:
        if self.vrrp_global_info['version'] != self.version:
            return True
    return False