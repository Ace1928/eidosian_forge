from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
from ansible.module_utils.basic import AnsibleModule
def generate_power_info_config(self):
    power_info_config = vim.host.AutoStartManager.AutoPowerInfo()
    power_info_config.key = self.vm_obj
    power_info_config.startAction = self.power_info['start_action']
    power_info_config.startDelay = self.power_info['start_delay']
    power_info_config.startOrder = self.power_info['start_order']
    power_info_config.stopAction = self.power_info['stop_action']
    power_info_config.stopDelay = self.power_info['stop_delay']
    power_info_config.waitForHeartbeat = self.power_info['wait_for_heartbeat']
    return power_info_config