from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible_collections.community.general.plugins.module_utils.opennebula import flatten, render
def resume_vm(module, client, vm):
    vm = client.vm.info(vm.ID)
    changed = False
    state = vm.STATE
    if state in [VM_STATES.index('HOLD')]:
        changed = release_vm(module, client, vm)
        return changed
    lcm_state = vm.LCM_STATE
    if lcm_state == LCM_STATES.index('SHUTDOWN_POWEROFF'):
        module.fail_json(msg="Cannot perform action 'resume' because this action is not available " + "for LCM_STATE: 'SHUTDOWN_POWEROFF'. Wait for the VM to shutdown properly")
    if lcm_state not in [LCM_STATES.index('RUNNING')]:
        changed = True
    if changed and (not module.check_mode):
        client.vm.action('resume', vm.ID)
    return changed