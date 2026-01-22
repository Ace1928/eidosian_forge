from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def check_dvs_pvlan(self):
    for pvlan in self.dv_switch.config.pvlanConfig:
        if pvlan.primaryVlanId == int(self.module.params['vlan_id']):
            return True
        if pvlan.secondaryVlanId == int(self.module.params['vlan_id']):
            return True
    return False