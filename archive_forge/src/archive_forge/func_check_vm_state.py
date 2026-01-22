from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, wait_for_task
from ansible.module_utils._text import to_native
def check_vm_state(self, vm_obj):
    """
        To add serial port, the VM must be in powered off state

        Input:
          - vm: Virtual Machine

        Output:
          - True if vm is in poweredOff state
          - module fails otherwise
        """
    if vm_obj.runtime.powerState == vim.VirtualMachinePowerState.poweredOff:
        return True
    self.module.fail_json(msg='A serial device cannot be added to a VM in the current state(' + vm_obj.runtime.powerState + '). Please use the vmware_guest_powerstate module to power off the VM')