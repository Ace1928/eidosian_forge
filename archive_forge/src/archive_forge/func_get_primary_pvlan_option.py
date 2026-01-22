from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def get_primary_pvlan_option(self, primary_vlan):
    """Get Primary PVLAN option"""
    primary_pvlan_id = primary_vlan.get('primary_pvlan_id', None)
    if primary_pvlan_id is None:
        self.module.fail_json(msg="Please specify primary_pvlan_id in primary_pvlans options as it's a required parameter")
    if primary_pvlan_id in (0, 4095):
        self.module.fail_json(msg='The VLAN IDs of 0 and 4095 are reserved and cannot be used as a primary PVLAN.')
    return primary_pvlan_id