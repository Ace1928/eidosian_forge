from __future__ import (absolute_import, division, print_function)
import re
import copy
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def check_vpn_target_para(self):
    """Check whether VPN target value is valid"""
    if self.route_distinguisher:
        if self.route_distinguisher.lower() != 'auto' and (not is_valid_value(self.route_distinguisher)):
            self.module.fail_json(msg='Error: Route distinguisher has invalid value %s.' % self.route_distinguisher)
    if self.vpn_target_export:
        for ele in self.vpn_target_export:
            if ele.lower() != 'auto' and (not is_valid_value(ele)):
                self.module.fail_json(msg='Error: VPN target extended community attribute has invalid value %s.' % ele)
    if self.vpn_target_import:
        for ele in self.vpn_target_import:
            if ele.lower() != 'auto' and (not is_valid_value(ele)):
                self.module.fail_json(msg='Error: VPN target extended community attribute has invalid value %s.' % ele)
    if self.vpn_target_both:
        for ele in self.vpn_target_both:
            if ele.lower() != 'auto' and (not is_valid_value(ele)):
                self.module.fail_json(msg='Error: VPN target extended community attribute has invalid value %s.' % ele)