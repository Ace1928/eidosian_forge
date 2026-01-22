from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware_sms import (
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec
from ansible.module_utils._text import to_native
def state_exit_unchanged(self):
    """
        Exit without any change
        """
    self.module.exit_json(changed=False)