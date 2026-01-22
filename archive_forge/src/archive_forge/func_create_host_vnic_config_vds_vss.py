from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def create_host_vnic_config_vds_vss(self):
    host_vnic_config = vim.host.VirtualNic.Config()
    host_vnic_config.spec = vim.host.VirtualNic.Specification()
    host_vnic_config.changeOperation = 'edit'
    host_vnic_config.device = self.device
    host_vnic_config.spec.portgroup = self.migrate_portgroup_name
    return host_vnic_config