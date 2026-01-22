from __future__ import absolute_import, division, print_function
import re
import time
import string
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.network import is_mac
from ansible.module_utils._text import to_text, to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible_collections.community.vmware.plugins.module_utils.vm_device_helper import PyVmomiDeviceHelper
from ansible_collections.community.vmware.plugins.module_utils.vmware_spbm import SPBM
def get_vm_scsi_controllers(self, vm_obj):
    scsi_ctls = []
    if vm_obj is None:
        return None
    for device in vm_obj.config.hardware.device:
        if self.device_helper.is_scsi_controller(device):
            scsi_ctl = vim.vm.device.VirtualDeviceSpec()
            scsi_ctl.device = device
            scsi_ctls.append(scsi_ctl)
    return scsi_ctls