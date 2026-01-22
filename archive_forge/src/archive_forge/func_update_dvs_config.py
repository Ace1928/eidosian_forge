from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def update_dvs_config(self, switch_object, spec):
    """Update DVS config"""
    try:
        task = switch_object.ReconfigureDvs_Task(spec)
        wait_for_task(task)
    except TaskError as invalid_argument:
        self.module.fail_json(msg='Failed to update DVS : %s' % to_native(invalid_argument))