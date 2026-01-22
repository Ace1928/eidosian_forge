from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def create_port_group_config(self):
    port_group_config = vim.host.PortGroup.Config()
    port_group_config.spec = vim.host.PortGroup.Specification()
    port_group_config.changeOperation = 'remove'
    port_group_config.spec.name = self.current_portgroup_name
    port_group_config.spec.vlanId = -1
    port_group_config.spec.vswitchName = self.current_switch_name
    port_group_config.spec.policy = vim.host.NetworkPolicy()
    return port_group_config