from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible.module_utils._text import to_native
def set_vsan_service_type(self, enable_vsan):
    """
        Set VSAN service type
        Returns: result of UpdateVsan_Task

        """
    result = None
    vsan_system = self.esxi_host_obj.configManager.vsanSystem
    vsan_system_config = vsan_system.config
    vsan_config = vim.vsan.host.ConfigInfo()
    vsan_config.networkInfo = vsan_system_config.networkInfo
    current_vsan_vnics = [portConfig.device for portConfig in vsan_system_config.networkInfo.port]
    changed = False
    result = '%s NIC %s (currently enabled NICs: %s) : ' % ('Enable' if enable_vsan else 'Disable', self.vnic.device, current_vsan_vnics)
    if not enable_vsan:
        if self.vnic.device in current_vsan_vnics:
            vsan_config.networkInfo.port = list(filter(lambda portConfig: portConfig.device != self.vnic.device, vsan_config.networkInfo.port))
            changed = True
    elif self.vnic.device not in current_vsan_vnics:
        vsan_port_config = vim.vsan.host.ConfigInfo.NetworkInfo.PortConfig()
        vsan_port_config.device = self.vnic.device
        if vsan_config.networkInfo is None:
            vsan_config.networkInfo = vim.vsan.host.ConfigInfo.NetworkInfo()
            vsan_config.networkInfo.port = [vsan_port_config]
        else:
            vsan_config.networkInfo.port.append(vsan_port_config)
        changed = True
    if not self.module.check_mode and changed:
        try:
            vsan_task = vsan_system.UpdateVsan_Task(vsan_config)
            task_result = wait_for_task(vsan_task)
            if task_result[0]:
                result += 'Success'
            else:
                result += 'Failed'
        except TaskError as task_err:
            self.module.fail_json(msg='Failed to set service type to vsan for %s : %s' % (self.vnic.device, to_native(task_err)))
    if self.module.check_mode:
        result += 'Dry-run'
    return result