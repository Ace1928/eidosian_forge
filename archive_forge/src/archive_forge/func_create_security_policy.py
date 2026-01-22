from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
from ansible.module_utils._text import to_native
def create_security_policy(self):
    """
        Create a Security Policy
        Returns: Security Policy object
        """
    security_policy = vim.host.NetworkPolicy.SecurityPolicy()
    security_policy.allowPromiscuous = self.sec_promiscuous_mode
    security_policy.macChanges = self.sec_mac_changes
    security_policy.forgedTransmits = self.sec_forged_transmits
    return security_policy