from __future__ import absolute_import, division, print_function
import re
from random import randint
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, \
from ansible_collections.community.vmware.plugins.module_utils.vm_device_helper import PyVmomiDeviceHelper
def reconfigure_vm(self, config_spec, device_type):
    """
        Reconfigure virtual machine after modifying device spec
        Args:
            config_spec: Config Spec
            device_type: Type of device being modified

        Returns: Boolean status 'changed' and actual task result

        """
    changed, results = (False, '')
    try:
        task = self.vm.ReconfigVM_Task(spec=config_spec)
        changed, results = wait_for_task(task)
    except vim.fault.InvalidDeviceSpec as invalid_device_spec:
        self.module.fail_json(msg="Failed to manage '%s' on given virtual machine due to invalid device spec : %s" % (device_type, to_native(invalid_device_spec.msg)), details='Please check ESXi server logs for more details.')
    except vim.fault.RestrictedVersion as e:
        self.module.fail_json(msg='Failed to reconfigure virtual machine due to product versioning restrictions: %s' % to_native(e.msg))
    return (changed, results)