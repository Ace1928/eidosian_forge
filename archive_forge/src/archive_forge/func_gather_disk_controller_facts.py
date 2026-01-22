from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, wait_for_task, TaskError
from ansible_collections.community.vmware.plugins.module_utils.vm_device_helper import PyVmomiDeviceHelper
def gather_disk_controller_facts(self):
    """
        Gather existing controller facts

        Return: A dictionary of each type controller facts
        """
    disk_ctl_facts = dict(scsi=dict(), sata=dict(), nvme=dict(), usb2=dict(), usb3=dict())
    for device in self.current_vm_obj.config.hardware.device:
        ctl_facts_dict = dict()
        if isinstance(device, tuple(self.controller_types.values())):
            ctl_facts_dict[device.busNumber] = dict(controller_summary=device.deviceInfo.summary, controller_label=device.deviceInfo.label, controller_busnumber=device.busNumber, controller_controllerkey=device.controllerKey, controller_devicekey=device.key, controller_unitnumber=device.unitNumber, controller_disks_devicekey=device.device)
            if hasattr(device, 'sharedBus'):
                ctl_facts_dict[device.busNumber]['controller_bus_sharing'] = device.sharedBus
            if isinstance(device, tuple(self.device_helper.scsi_device_type.values())):
                disk_ctl_facts['scsi'].update(ctl_facts_dict)
            if isinstance(device, self.device_helper.nvme_device_type):
                disk_ctl_facts['nvme'].update(ctl_facts_dict)
            if isinstance(device, self.device_helper.sata_device_type):
                disk_ctl_facts['sata'].update(ctl_facts_dict)
            if isinstance(device, self.device_helper.usb_device_type.get('usb2')):
                disk_ctl_facts['usb2'].update(ctl_facts_dict)
            if isinstance(device, self.device_helper.usb_device_type.get('usb3')):
                disk_ctl_facts['usb3'].update(ctl_facts_dict)
    return disk_ctl_facts