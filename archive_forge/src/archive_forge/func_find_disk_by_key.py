from __future__ import absolute_import, division, print_function
import re
from random import randint
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, \
from ansible_collections.community.vmware.plugins.module_utils.vm_device_helper import PyVmomiDeviceHelper
def find_disk_by_key(self, disk_key, disk_unit_number):
    found_disk = None
    for device in self.vm.config.hardware.device:
        if isinstance(device, vim.vm.device.VirtualDisk) and device.key == disk_key:
            if device.unitNumber == disk_unit_number:
                found_disk = device
                break
    return found_disk