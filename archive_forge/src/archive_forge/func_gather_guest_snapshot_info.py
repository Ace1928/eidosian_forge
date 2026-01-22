from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, list_snapshots, vmware_argument_spec
@staticmethod
def gather_guest_snapshot_info(vm_obj=None):
    """
        Return snapshot related information about given virtual machine
        Args:
            vm_obj: Virtual Machine Managed object

        Returns: Dictionary containing snapshot information

        """
    if vm_obj is None:
        return {}
    return list_snapshots(vm=vm_obj)