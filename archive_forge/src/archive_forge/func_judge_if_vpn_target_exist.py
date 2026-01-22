from __future__ import (absolute_import, division, print_function)
import re
import copy
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def judge_if_vpn_target_exist(self, vpn_target_type):
    """Judge whether proposed vpn target has existed"""
    vpn_target = list()
    if vpn_target_type == 'vpn_target_import':
        vpn_target.extend(self.existing['vpn_target_both'])
        vpn_target.extend(self.existing['vpn_target_import'])
        return set(self.proposed['vpn_target_import']).issubset(vpn_target)
    elif vpn_target_type == 'vpn_target_export':
        vpn_target.extend(self.existing['vpn_target_both'])
        vpn_target.extend(self.existing['vpn_target_export'])
        return set(self.proposed['vpn_target_export']).issubset(vpn_target)
    return False