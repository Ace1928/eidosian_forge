from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def create_ldp_spec(self):
    """Create Link Discovery Protocol config spec"""
    ldp_config_spec = vim.host.LinkDiscoveryProtocolConfig()
    if self.discovery_protocol == 'disabled':
        ldp_config_spec.protocol = 'cdp'
        ldp_config_spec.operation = 'none'
    else:
        ldp_config_spec.protocol = self.discovery_protocol
        ldp_config_spec.operation = self.discovery_operation
    return ldp_config_spec