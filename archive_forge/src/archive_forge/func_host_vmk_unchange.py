from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible.module_utils._text import to_native
def host_vmk_unchange(self):
    """
        Denote no change in VMKernel
        Returns: NA

        """
    self.module.exit_json(changed=False)