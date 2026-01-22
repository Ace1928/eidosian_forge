from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def manage_drs_group_members(self):
    """
        Add a DRS host/vm group members
        """
    if self._vm_list is None:
        self._manage_host_group()
    elif self._host_list is None:
        self._manage_vm_group()
    else:
        self.module.fail_json(msg='Failed, no hosts or vms defined')