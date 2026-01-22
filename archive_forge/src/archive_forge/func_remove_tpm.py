from __future__ import absolute_import, division, print_function
import traceback
from random import randint
from ansible.module_utils.common.network import is_mac
from ansible.module_utils.basic import missing_required_lib
def remove_tpm(self, vtpm_device):
    vtpm_device_spec = vim.vm.device.VirtualDeviceSpec()
    vtpm_device_spec.operation = vim.vm.device.VirtualDeviceSpec.Operation.remove
    vtpm_device_spec.device = vtpm_device
    return vtpm_device_spec