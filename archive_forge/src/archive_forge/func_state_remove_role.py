from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
def state_remove_role(self):
    """Remove local role"""
    results = dict()
    results['role_name'] = self.role_name
    results['role_id'] = self.current_role.roleId
    results['local_role_name'] = self.role_name
    if self.module.check_mode:
        results['msg'] = 'Role would be deleted'
    else:
        try:
            self.content.authorizationManager.RemoveAuthorizationRole(roleId=self.current_role.roleId, failIfUsed=self.force)
            results['msg'] = 'Role deleted'
        except vim.fault.NotFound as not_found:
            self.module.fail_json(msg='Failed to remove a role %s as the user specified role name does not exist.' % self.role_name, details=not_found.msg)
        except vim.fault.RemoveFailed as remove_failed:
            msg = "Failed to remove role '%s' as the user specified role name." % self.role_name
            if self.force:
                msg += ' Use force_remove as True.'
            self.module.fail_json(msg=msg, details=remove_failed.msg)
        except vmodl.fault.InvalidArgument as invalid_argument:
            self.module.fail_json(msg='Failed to remove a role %s as the user specified role is a system role' % self.role_name, details=invalid_argument.msg)
    self.module.exit_json(changed=True, result=results)