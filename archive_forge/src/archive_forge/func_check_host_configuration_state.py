from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
from ansible.module_utils._text import to_native
def check_host_configuration_state(self):
    change_list = []
    for host in self.hosts:
        changed = False
        msg = ''
        self.results[host.name] = dict()
        if host.runtime.connectionState == 'connected':
            host_kernel_manager = host.configManager.kernelModuleSystem
            if host_kernel_manager:
                original_options = self.get_kernel_module_option(host, self.kernel_module_name)
                desired_options = self.kernel_module_option
                if original_options != desired_options:
                    changed = True
                    if self.module.check_mode:
                        msg = 'Options would be changed on the kernel module'
                    else:
                        self.apply_kernel_module_option(host, self.kernel_module_name, desired_options)
                        msg = 'Options have been changed on the kernel module'
                        self.results[host.name]['configured_options'] = desired_options
                else:
                    msg = 'Options are already the same'
                change_list.append(changed)
                self.results[host.name]['changed'] = changed
                self.results[host.name]['msg'] = msg
                self.results[host.name]['original_options'] = original_options
            else:
                msg = 'No kernel module manager found on host %s - impossible to configure.' % host.name
                self.results[host.name]['changed'] = changed
                self.results[host.name]['msg'] = msg
        else:
            msg = 'Host %s is disconnected and cannot be changed.' % host.name
            self.results[host.name]['changed'] = changed
            self.results[host.name]['msg'] = msg
    self.module.exit_json(changed=any(change_list), host_kernel_status=self.results)