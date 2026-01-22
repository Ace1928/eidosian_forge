from __future__ import (absolute_import, division, print_function)
import re
import copy
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def is_vpn_target_exist(self, target_type, value):
    """Judge whether VPN target has existed"""
    if target_type == 'export_extcommunity':
        if value not in self.existing['vpn_target_export'] and value not in self.existing['vpn_target_both']:
            return False
        return True
    if target_type == 'import_extcommunity':
        if value not in self.existing['vpn_target_import'] and value not in self.existing['vpn_target_both']:
            return False
        return True
    return False