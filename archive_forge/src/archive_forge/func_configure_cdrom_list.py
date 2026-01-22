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
def configure_cdrom_list(self, vm_obj):
    configured_cdroms = self.sanitize_cdrom_params()
    cdrom_devices = self.get_vm_cdrom_devices(vm=vm_obj)
    ide_devices = self.get_vm_ide_devices(vm=vm_obj)
    sata_devices = self.get_vm_sata_devices(vm=vm_obj)
    for expected_cdrom_spec in configured_cdroms:
        ctl_device = None
        if expected_cdrom_spec['ctl_type'] == 'ide' and ide_devices:
            for device in ide_devices:
                if device.busNumber == expected_cdrom_spec['ctl_num']:
                    ctl_device = device
                    break
        if expected_cdrom_spec['ctl_type'] == 'sata' and sata_devices:
            for device in sata_devices:
                if device.busNumber == expected_cdrom_spec['ctl_num']:
                    ctl_device = device
                    break
        if not ctl_device:
            if expected_cdrom_spec['ctl_type'] == 'ide':
                ide_ctl = self.device_helper.create_ide_controller(bus_number=expected_cdrom_spec['ctl_num'])
                ctl_device = ide_ctl.device
                self.change_detected = True
                self.configspec.deviceChange.append(ide_ctl)
            if expected_cdrom_spec['ctl_type'] == 'sata':
                sata_ctl = self.device_helper.create_sata_controller(bus_number=expected_cdrom_spec['ctl_num'])
                ctl_device = sata_ctl.device
                self.change_detected = True
                self.configspec.deviceChange.append(sata_ctl)
        for cdrom in expected_cdrom_spec['cdroms']:
            cdrom_device = None
            iso_path = cdrom.get('iso_path')
            unit_number = cdrom.get('unit_number')
            for target_cdrom in cdrom_devices:
                if target_cdrom.controllerKey == ctl_device.key and target_cdrom.unitNumber == unit_number:
                    cdrom_device = target_cdrom
                    break
            if not cdrom_device and cdrom.get('state') != 'absent':
                if vm_obj and vm_obj.runtime.powerState == vim.VirtualMachinePowerState.poweredOn and isinstance(ctl_device, vim.vm.device.VirtualIDEController) and (not self.module.check_mode):
                    self.module.fail_json(msg='CD-ROM attach to IDE controller not support hot-add.')
                if len(ctl_device.device) == 2 and isinstance(ctl_device, vim.vm.device.VirtualIDEController):
                    self.module.fail_json(msg='Maximum number of CD-ROMs attached to IDE controller is 2.')
                if len(ctl_device.device) == 30 and isinstance(ctl_device, vim.vm.device.VirtualAHCIController):
                    self.module.fail_json(msg='Maximum number of CD-ROMs attached to SATA controller is 30.')
                cdrom_spec = self.device_helper.create_cdrom(ctl_device=ctl_device, cdrom_type=cdrom['type'], iso_path=iso_path, unit_number=unit_number)
                self.change_detected = True
                self.configspec.deviceChange.append(cdrom_spec)
            elif cdrom_device and cdrom.get('state') != 'absent' and (not self.device_helper.is_equal_cdrom(vm_obj=vm_obj, cdrom_device=cdrom_device, cdrom_type=cdrom['type'], iso_path=iso_path)):
                self.device_helper.update_cdrom_config(vm_obj, cdrom, cdrom_device, iso_path=iso_path)
                cdrom_spec = vim.vm.device.VirtualDeviceSpec()
                cdrom_spec.operation = vim.vm.device.VirtualDeviceSpec.Operation.edit
                cdrom_spec.device = cdrom_device
                self.change_detected = True
                self.configspec.deviceChange.append(cdrom_spec)
            elif cdrom_device and cdrom.get('state') == 'absent':
                if vm_obj and vm_obj.runtime.powerState != vim.VirtualMachinePowerState.poweredOff and isinstance(ctl_device, vim.vm.device.VirtualIDEController) and (not self.module.check_mode):
                    self.module.fail_json(msg='CD-ROM attach to IDE controller not support hot-remove.')
                cdrom_spec = self.device_helper.remove_cdrom(cdrom_device)
                self.change_detected = True
                self.configspec.deviceChange.append(cdrom_spec)