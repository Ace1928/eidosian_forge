from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, wait_for_task, TaskError
from ansible_collections.community.vmware.plugins.module_utils.vm_device_helper import PyVmomiDeviceHelper
def create_controller(self, ctl_type, bus_sharing, bus_number=0):
    """
        Create new disk or USB controller with specified type
        Args:
            ctl_type: controller type
            bus_number: disk controller bus number
            bus_sharing: noSharing, virtualSharing, physicalSharing

        Return: Virtual device spec for virtual controller
        """
    if ctl_type == 'sata' or ctl_type == 'nvme' or ctl_type in self.device_helper.scsi_device_type.keys():
        disk_ctl = self.device_helper.create_disk_controller(ctl_type, bus_number, bus_sharing)
    elif ctl_type in self.device_helper.usb_device_type.keys():
        disk_ctl = vim.vm.device.VirtualDeviceSpec()
        disk_ctl.operation = vim.vm.device.VirtualDeviceSpec.Operation.add
        disk_ctl.device = self.device_helper.usb_device_type.get(ctl_type)()
        if ctl_type == 'usb2':
            disk_ctl.device.key = 7000
        elif ctl_type == 'usb3':
            disk_ctl.device.key = 14000
        disk_ctl.device.deviceInfo = vim.Description()
        disk_ctl.device.busNumber = bus_number
    return disk_ctl