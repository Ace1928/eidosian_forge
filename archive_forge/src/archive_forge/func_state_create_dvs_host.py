from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible.module_utils._text import to_native
def state_create_dvs_host(self):
    operation, changed, result = ('add', True, None)
    if not self.module.check_mode:
        changed, result = self.modify_dvs_host(operation)
        if changed:
            self.set_desired_state()
            changed, result = self.modify_dvs_host('edit')
        else:
            self.module.exit_json(changed=changed, result=to_native(result))
    self.module.exit_json(changed=changed, result=to_native(result))