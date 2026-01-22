from __future__ import (absolute_import, division, print_function)
import re
import copy
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def check_undo_params_if_exist(self):
    """Check whether all undo parameters is existed"""
    if self.vpn_target_import:
        for ele in self.vpn_target_import:
            if ele not in self.evpn_info['vpn_target_import'] and ele not in self.evpn_info['vpn_target_both']:
                self.module.fail_json(msg='Error: VPN target import attribute value %s does not exist.' % ele)
    if self.vpn_target_export:
        for ele in self.vpn_target_export:
            if ele not in self.evpn_info['vpn_target_export'] and ele not in self.evpn_info['vpn_target_both']:
                self.module.fail_json(msg='Error: VPN target export attribute value %s does not exist.' % ele)
    if self.vpn_target_both:
        for ele in self.vpn_target_both:
            if ele not in self.evpn_info['vpn_target_both']:
                self.module.fail_json(msg='Error: VPN target export and import attribute value %s does not exist.' % ele)