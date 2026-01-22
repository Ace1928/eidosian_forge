from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def get_vm_port(self, vm_name, nic_label):
    """Finds the port of the VM
        Returns
        -------
        str
            the port number as a string, or None if the NIC couldnt be found
        """
    vm = find_vm_by_name(self.content, vm_name)
    if vm is None:
        self.module.fail_json(msg='There is no VM with the name: {0:s}.'.format(vm_name))
    for hardware in vm.config.hardware.device:
        if isinstance(hardware, vim.vm.device.VirtualVmxnet3):
            if hardware.deviceInfo.label == nic_label:
                return hardware.backing.port.portKey
    return None