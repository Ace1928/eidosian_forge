from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible.module_utils._text import to_native
def host_vmk_delete(self):
    """
        Delete VMKernel
        Returns: NA

        """
    results = dict(changed=False, msg='')
    vmk_device = self.vnic.device
    try:
        if self.module.check_mode:
            results['msg'] = 'VMkernel Adapter would be deleted'
        else:
            self.esxi_host_obj.configManager.networkSystem.RemoveVirtualNic(vmk_device)
            results['msg'] = 'VMkernel Adapter deleted'
        results['changed'] = True
        results['device'] = vmk_device
    except vim.fault.NotFound as not_found:
        self.module.fail_json(msg='Failed to find vmk to delete due to %s' % to_native(not_found.msg))
    except vim.fault.HostConfigFault as host_config_fault:
        self.module.fail_json(msg='Failed to delete vmk due host config issues : %s' % to_native(host_config_fault.msg))
    self.module.exit_json(**results)