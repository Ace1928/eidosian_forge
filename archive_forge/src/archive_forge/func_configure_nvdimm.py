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
def configure_nvdimm(self, vm_obj):
    """
        Manage virtual NVDIMM device to the virtual machine
        Args:
            vm_obj: virtual machine object
        """
    if self.params['nvdimm']['state']:
        if self.params['nvdimm']['state'] == 'absent' and (not self.params['nvdimm']['label']):
            self.module.fail_json(msg="Please specify the label of virtual NVDIMM device using 'label' parameter when state is set to 'absent'.")
        if vm_obj and (not vm_obj.config.template):
            if vm_obj.runtime.powerState != vim.VirtualMachinePowerState.poweredOff and (not self.module.check_mode):
                self.module.fail_json(msg='VM is not in power off state, can not do virtual NVDIMM configuration.')
        nvdimm_ctl_exists = False
        if vm_obj and (not vm_obj.config.template):
            nvdimm_ctl = self.get_vm_nvdimm_ctl_device(vm=vm_obj)
            if len(nvdimm_ctl) != 0:
                nvdimm_ctl_exists = True
                nvdimm_ctl_key = nvdimm_ctl[0].key
                if self.params['nvdimm']['label'] is not None:
                    nvdimm_devices = self.get_vm_nvdimm_devices(vm=vm_obj)
                    if len(nvdimm_devices) != 0:
                        existing_nvdimm_dev = self.device_helper.find_nvdimm_by_label(nvdimm_label=self.params['nvdimm']['label'], nvdimm_devices=nvdimm_devices)
                        if existing_nvdimm_dev is not None:
                            if self.params['nvdimm']['state'] == 'absent':
                                nvdimm_remove_spec = self.device_helper.remove_nvdimm(nvdimm_device=existing_nvdimm_dev)
                                self.change_detected = True
                                self.configspec.deviceChange.append(nvdimm_remove_spec)
                            elif existing_nvdimm_dev.capacityInMB < self.params['nvdimm']['size_mb']:
                                nvdimm_config_spec = self.device_helper.update_nvdimm_config(nvdimm_device=existing_nvdimm_dev, nvdimm_size=self.params['nvdimm']['size_mb'])
                                self.change_detected = True
                                self.configspec.deviceChange.append(nvdimm_config_spec)
                            elif existing_nvdimm_dev.capacityInMB > self.params['nvdimm']['size_mb']:
                                self.module.fail_json(msg='Can not change NVDIMM device size to %s MB, which is smaller than the current size %s MB.' % (self.params['nvdimm']['size_mb'], existing_nvdimm_dev.capacityInMB))
        if vm_obj is None or (vm_obj and (not vm_obj.config.template) and (self.params['nvdimm']['label'] is None)):
            if self.params['nvdimm']['state'] == 'present':
                vc_pmem_profile_id = None
                if self.is_vcenter():
                    storage_profile_name = 'Host-local PMem Default Storage Policy'
                    spbm = SPBM(self.module)
                    pmem_profile = spbm.find_storage_profile_by_name(profile_name=storage_profile_name)
                    if pmem_profile is None:
                        self.module.fail_json(msg="Can not find PMem storage policy with name '%s'." % storage_profile_name)
                    vc_pmem_profile_id = pmem_profile.profileId.uniqueId
                if not nvdimm_ctl_exists:
                    nvdimm_ctl_spec = self.device_helper.create_nvdimm_controller()
                    self.configspec.deviceChange.append(nvdimm_ctl_spec)
                    nvdimm_ctl_key = nvdimm_ctl_spec.device.key
                nvdimm_dev_spec = self.device_helper.create_nvdimm_device(nvdimm_ctl_dev_key=nvdimm_ctl_key, pmem_profile_id=vc_pmem_profile_id, nvdimm_dev_size_mb=self.params['nvdimm']['size_mb'])
                self.change_detected = True
                self.configspec.deviceChange.append(nvdimm_dev_spec)