from __future__ import absolute_import, division, print_function
import traceback
from random import randint
from ansible.module_utils.common.network import is_mac
from ansible.module_utils.basic import missing_required_lib
@staticmethod
def update_cdrom_config(vm_obj, cdrom_spec, cdrom_device, iso_path=None):
    if cdrom_spec['type'] in ['client', 'none']:
        cdrom_device.backing = vim.vm.device.VirtualCdrom.RemotePassthroughBackingInfo()
    elif cdrom_spec['type'] == 'iso' and iso_path is not None:
        cdrom_device.backing = vim.vm.device.VirtualCdrom.IsoBackingInfo(fileName=iso_path)
    cdrom_device.connectable = vim.vm.device.VirtualDevice.ConnectInfo()
    cdrom_device.connectable.allowGuestControl = True
    cdrom_device.connectable.startConnected = cdrom_spec['type'] != 'none'
    if vm_obj and vm_obj.runtime.powerState == vim.VirtualMachinePowerState.poweredOn:
        cdrom_device.connectable.connected = cdrom_spec['type'] != 'none'