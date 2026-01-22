from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
@staticmethod
def create_pvlan_config_spec(operation, primary_pvlan_id, secondary_pvlan_id, pvlan_type):
    """
            Create PVLAN config spec
            operation: add, edit, or remove
            Returns: PVLAN config spec
        """
    pvlan_spec = vim.dvs.VmwareDistributedVirtualSwitch.PvlanConfigSpec()
    pvlan_spec.operation = operation
    pvlan_spec.pvlanEntry = vim.dvs.VmwareDistributedVirtualSwitch.PvlanMapEntry()
    pvlan_spec.pvlanEntry.primaryVlanId = primary_pvlan_id
    pvlan_spec.pvlanEntry.secondaryVlanId = secondary_pvlan_id
    pvlan_spec.pvlanEntry.pvlanType = pvlan_type
    return pvlan_spec