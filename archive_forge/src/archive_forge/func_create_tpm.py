from __future__ import absolute_import, division, print_function
import traceback
from random import randint
from ansible.module_utils.common.network import is_mac
from ansible.module_utils.basic import missing_required_lib
def create_tpm(self):
    vtpm_device_spec = vim.vm.device.VirtualDeviceSpec()
    vtpm_device_spec.operation = vim.vm.device.VirtualDeviceSpec.Operation.add
    vtpm_device_spec.device = vim.vm.device.VirtualTPM()
    vtpm_device_spec.device.deviceInfo = vim.Description()
    return vtpm_device_spec