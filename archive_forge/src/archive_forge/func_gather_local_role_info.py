from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
def gather_local_role_info(self):
    """Gather info about local roles"""
    results = list()
    for role in self.content.authorizationManager.roleList:
        results.append(dict(role_name=role.name, role_id=role.roleId, privileges=list(role.privilege), role_system=role.system, role_info_label=role.info.label, role_info_summary=role.info.summary))
    self.module.exit_json(changed=False, local_role_info=results)