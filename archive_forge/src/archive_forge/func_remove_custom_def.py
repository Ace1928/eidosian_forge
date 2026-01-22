from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
def remove_custom_def(self, field):
    changed = False
    for x in self.custom_field_mgr:
        if x.name == field and x.managedObjectType == self.object_type:
            changed = True
            if not self.module.check_mode:
                self.content.customFieldsManager.RemoveCustomFieldDef(key=x.key)
            break
    return {'changed': changed, 'failed': False}