from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
from ansible.module_utils._text import to_native
def get_kernel_module_option(self, host, kmod_name):
    host_kernel_manager = host.configManager.kernelModuleSystem
    try:
        return host_kernel_manager.QueryConfiguredModuleOptionString(self.kernel_module_name)
    except vim.fault.NotFound as kernel_fault:
        self.module.fail_json(msg="Failed to find kernel module on host '%s'. More information: %s" % (host.name, to_native(kernel_fault.msg)))