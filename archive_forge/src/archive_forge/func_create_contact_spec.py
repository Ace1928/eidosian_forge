from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def create_contact_spec(self):
    """Create contact info spec"""
    contact_info_spec = vim.DistributedVirtualSwitch.ContactInfo()
    contact_info_spec.name = self.contact_name
    contact_info_spec.contact = self.contact_details
    return contact_info_spec