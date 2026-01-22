from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def is_dfs_group_info_change(self):
    """whether dfs group info"""
    if not self.dfs_group_info:
        return False
    if self.priority_id and self.dfs_group_info['priority'] != self.priority_id:
        return True
    if self.ip_address and self.dfs_group_info['ipAddress'] != self.ip_address:
        return True
    if self.vpn_instance_name and self.dfs_group_info['srcVpnName'] != self.vpn_instance_name:
        return True
    if self.nickname and self.dfs_group_info['localNickname'] != self.nickname:
        return True
    if self.pseudo_nickname and self.dfs_group_info['pseudoNickname'] != self.pseudo_nickname:
        return True
    if self.pseudo_priority and self.dfs_group_info['pseudoPriority'] != self.pseudo_priority:
        return True
    return False