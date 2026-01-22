from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def state_update_nioc_version(self):
    self.result['changed'] = True
    if not self.module.check_mode:
        self.set_nioc_version()
        self.result['dvswitch_nioc_status'] = 'Set NIOC to version %s' % self.version
        if self.check_resources() == 'update':
            self.set_nioc_resources(self.resource_changes)