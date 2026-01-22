from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, find_obj
def get_perms(self):
    self.current_perms = self.auth_manager.RetrieveEntityPermissions(self.current_obj, False)