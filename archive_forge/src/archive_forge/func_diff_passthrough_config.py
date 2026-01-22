from __future__ import absolute_import, division, print_function
import copy
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
from ansible.module_utils.basic import AnsibleModule
def diff_passthrough_config(self):
    """
        Check there are differences between a new and existing config each ESXi host.
        """
    self.diff_config = dict(before={}, after={})
    self.change_flag = False
    self.host_target_device_to_change_configuration = {}
    state = True if self.state == 'present' else False
    for host_has_checked_pci_devices in self.existent_devices:
        for esxi_hostname, value in host_has_checked_pci_devices.items():
            for key in ('before', 'after'):
                self.diff_config[key][esxi_hostname] = []
            self.host_target_device_to_change_configuration[esxi_hostname] = {'host_obj': None, 'new_configs': []}
            for target_device in self.devices:
                device = target_device['device']
                for checked_pci_device in value['checked_pci_devices']:
                    if device == checked_pci_device['device_name'] or device == checked_pci_device['device_id']:
                        before = dict(checked_pci_device)
                        after = dict(copy.deepcopy(checked_pci_device))
                        if state != checked_pci_device['passthruEnabled']:
                            self.change_flag = True
                            after['passthruEnabled'] = state
                            self.host_target_device_to_change_configuration[esxi_hostname]['new_configs'].append(after)
                        self.host_target_device_to_change_configuration[esxi_hostname]['host_obj'] = value['host_obj']
                        self.diff_config['before'][esxi_hostname].append(before)
                        self.diff_config['after'][esxi_hostname].append(after)
            self.diff_config['before'][esxi_hostname] = sorted(self.de_duplication(self.diff_config['before'][esxi_hostname]), key=lambda d: d['device_name'])
            self.diff_config['after'][esxi_hostname] = sorted(self.de_duplication(self.diff_config['after'][esxi_hostname]), key=lambda d: d['device_name'])