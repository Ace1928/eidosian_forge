from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def state_update_nioc_resources(self):
    self.result['changed'] = True
    if not self.module.check_mode:
        self.result['dvswitch_nioc_status'] = 'Resource configuration modified'
        self.set_nioc_resources(self.resource_changes)