from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, find_obj
def setup_permission(self):
    perm = vim.AuthorizationManager.Permission()
    perm.entity = self.current_obj
    perm.group = self.is_group
    perm.principal = self.applied_to
    perm.roleId = self.role.roleId
    perm.propagate = self.params['recursive']
    return perm