from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def state_disable_evc(self):
    """
        Disable EVC Mode
        """
    changed, result = (False, None)
    try:
        if not self.module.check_mode:
            evc_task = self.evcm.DisableEvcMode_Task()
            changed, result = wait_for_task(evc_task)
        if self.module.check_mode:
            changed = True
        self.module.exit_json(changed=changed, msg="EVC Mode has been disabled on cluster '%s'." % self.cluster_name)
    except TaskError as invalid_argument:
        self.module.fail_json(msg='Failed to disable EVC mode: %s' % to_native(invalid_argument))