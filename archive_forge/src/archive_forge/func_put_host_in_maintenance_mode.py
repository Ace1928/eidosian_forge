from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def put_host_in_maintenance_mode(self, host_object):
    """Put host in maintenance mode, if not already"""
    if not host_object.runtime.inMaintenanceMode:
        try:
            try:
                maintenance_mode_task = host_object.EnterMaintenanceMode_Task(300, True, None)
            except vim.fault.InvalidState as invalid_state:
                self.module.fail_json(msg='The host is already in maintenance mode : %s' % to_native(invalid_state))
            except vim.fault.Timedout as timed_out:
                self.module.fail_json(msg='The maintenance mode operation timed out : %s' % to_native(timed_out))
            except vim.fault.Timedout as timed_out:
                self.module.fail_json(msg='The maintenance mode operation was canceled : %s' % to_native(timed_out))
            wait_for_task(maintenance_mode_task)
        except TaskError as task_err:
            self.module.fail_json(msg='Failed to put the host in maintenance mode : %s' % to_native(task_err))