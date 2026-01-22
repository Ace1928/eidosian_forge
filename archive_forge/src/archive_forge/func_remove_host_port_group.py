from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
from ansible.module_utils._text import to_native
def remove_host_port_group(self, host_system):
    """
        Remove a Port Group from a given host
        Args:
            host_system: Name of Host System
        """
    host_results = dict(changed=False, msg='')
    if self.module.check_mode:
        host_results['msg'] = 'Port Group would be removed'
    else:
        try:
            host_system.configManager.networkSystem.RemovePortGroup(pgName=self.portgroup)
            host_results['msg'] = 'Port Group removed'
        except vim.fault.NotFound as not_found:
            self.module.fail_json(msg='Failed to remove Portgroup as it was not found: %s' % to_native(not_found.msg))
        except vim.fault.ResourceInUse as resource_in_use:
            self.module.fail_json(msg='Failed to remove Portgroup as it is in use: %s' % to_native(resource_in_use.msg))
        except vim.fault.HostConfigFault as host_config_fault:
            self.module.fail_json(msg='Failed to remove Portgroup due to configuration failures: %s' % to_native(host_config_fault.msg))
    host_results['changed'] = True
    host_results['portgroup'] = self.portgroup
    host_results['vswitch'] = self.switch
    return (True, host_results)