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
def configure_hardware_params(self, vm_obj):
    """
        Function to configure hardware related configuration of virtual machine
        Args:
            vm_obj: virtual machine object
        """
    max_connections = self.params['hardware']['max_connections']
    if max_connections is not None:
        if vm_obj is None or max_connections != vm_obj.config.maxMksConnections:
            self.change_detected = True
            self.configspec.maxMksConnections = max_connections
    nested_virt = self.params['hardware']['nested_virt']
    if nested_virt is not None:
        if vm_obj is None or nested_virt != bool(vm_obj.config.nestedHVEnabled):
            self.change_detected = True
            self.configspec.nestedHVEnabled = nested_virt
    temp_version = self.params['hardware']['version']
    if temp_version is not None:
        new_version = None
        if temp_version.lower() == 'latest':
            if vm_obj and (not vm_obj.config.template):
                config_option_descriptors = vm_obj.environmentBrowser.QueryConfigOptionDescriptor()
                available_hw_versions = [int(option_desc.key.split('-')[1]) for option_desc in config_option_descriptors if option_desc.upgradeSupported]
                temp_version = max(available_hw_versions)
        else:
            try:
                temp_version = int(temp_version)
            except ValueError:
                self.module.fail_json(msg="Failed to set hardware.version '%s' value as valid values are either 'latest' or a number. Please check VMware documentation for valid VM hardware versions." % temp_version)
        if isinstance(temp_version, int):
            new_version = 'vmx-%02d' % temp_version
        if vm_obj is None:
            self.change_detected = True
            self.configspec.version = new_version
        elif not vm_obj.config.template:
            current_version = vm_obj.config.version
            version_digit = int(current_version.split('-', 1)[-1])
            if temp_version < version_digit:
                self.module.fail_json(msg="Current hardware version '%d' which is greater than the specified version '%d'. Downgrading hardware version is not supported. Please specify version greater than the current version." % (version_digit, temp_version))
            elif temp_version > version_digit:
                self.change_detected = True
                self.tracked_changes['hardware.version'] = temp_version
                self.configspec.version = new_version
                if not self.module.check_mode:
                    task = vm_obj.UpgradeVM_Task(new_version)
                    self.wait_for_task(task)
                    if task.info.state == 'error':
                        return {'changed': self.change_applied, 'failed': True, 'msg': task.info.error.msg, 'op': 'upgrade'}
                    self.change_applied = True
    secure_boot = self.params['hardware']['secure_boot']
    if secure_boot is not None:
        if vm_obj is None or secure_boot != vm_obj.config.bootOptions.efiSecureBootEnabled:
            self.change_detected = True
            self.configspec.bootOptions = vim.vm.BootOptions()
            self.configspec.bootOptions.efiSecureBootEnabled = secure_boot
    iommu = self.params['hardware']['iommu']
    if iommu is not None:
        if vm_obj is None or iommu != vm_obj.config.flags.vvtdEnabled:
            self.change_detected = True
            if self.configspec.flags is None:
                self.configspec.flags = vim.vm.FlagInfo()
            self.configspec.flags.vvtdEnabled = iommu
    virt_based_security = self.params['hardware']['virt_based_security']
    if virt_based_security is not None:
        if vm_obj is None or virt_based_security != vm_obj.config.flags.vbsEnabled:
            self.change_detected = True
            if self.configspec.flags is None:
                self.configspec.flags = vim.vm.FlagInfo()
            self.configspec.flags.vbsEnabled = virt_based_security