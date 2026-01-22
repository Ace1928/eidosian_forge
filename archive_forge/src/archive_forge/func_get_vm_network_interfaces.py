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
def get_vm_network_interfaces(self, vm=None):
    device_list = []
    if vm is None:
        return device_list
    for device in vm.config.hardware.device:
        for device_type in self.device_helper.nic_device_type.values():
            if isinstance(device, device_type):
                device_list.append(device)
    return device_list