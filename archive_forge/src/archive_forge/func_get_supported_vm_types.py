import socket
from oslo_log import log as logging
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils.winapi import libs as w_lib
def get_supported_vm_types(self):
    """Get the supported Hyper-V VM generations.

        Hyper-V Generation 2 VMs are supported in Windows 8.1,
        Windows Server / Hyper-V Server 2012 R2 or newer.

        :returns: array of supported VM generations (ex. ['hyperv-gen1'])
        """
    if self.check_min_windows_version(6, 3):
        return [constants.IMAGE_PROP_VM_GEN_1, constants.IMAGE_PROP_VM_GEN_2]
    else:
        return [constants.IMAGE_PROP_VM_GEN_1]