from __future__ import absolute_import, division, print_function
import traceback
from random import randint
from ansible.module_utils.common.network import is_mac
from ansible.module_utils.basic import missing_required_lib
@staticmethod
def is_nvdimm_controller(device):
    return isinstance(device, vim.vm.device.VirtualNVDIMMController)