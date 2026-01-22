from __future__ import absolute_import, division, print_function
import atexit
import time
import re
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils.ansible_release import __version__ as ANSIBLE_VERSION
def set_vm_power_state(module, vm_ref, power_state, timeout=300):
    """Controls VM power state.

    Args:
        module: Reference to Ansible module object.
        vm_ref (str): XAPI reference to VM.
        power_state (str): Power state to put VM into. Accepted values:

            - poweredon
            - poweredoff
            - restarted
            - suspended
            - shutdownguest
            - rebootguest

        timeout (int): timeout in seconds (default: 300).

    Returns:
        tuple (bool, str): Bool element is True if VM power state has
        changed by calling this function, else False. Str element carries
        a value of resulting power state as defined by XAPI - 'running',
        'halted' or 'suspended'.
    """
    if not vm_ref or vm_ref == 'OpaqueRef:NULL':
        module.fail_json(msg='Cannot set VM power state. Invalid VM reference supplied!')
    xapi_session = XAPI.connect(module)
    power_state = power_state.replace('_', '').replace('-', '').lower()
    vm_power_state_resulting = module_to_xapi_vm_power_state(power_state)
    state_changed = False
    try:
        vm_power_state_current = xapi_to_module_vm_power_state(xapi_session.xenapi.VM.get_power_state(vm_ref).lower())
        if vm_power_state_current != power_state:
            if power_state == 'poweredon':
                if not module.check_mode:
                    if vm_power_state_current == 'poweredoff':
                        xapi_session.xenapi.VM.start(vm_ref, False, False)
                    elif vm_power_state_current == 'suspended':
                        xapi_session.xenapi.VM.resume(vm_ref, False, False)
                    elif vm_power_state_current == 'paused':
                        xapi_session.xenapi.VM.unpause(vm_ref)
            elif power_state == 'poweredoff':
                if not module.check_mode:
                    xapi_session.xenapi.VM.hard_shutdown(vm_ref)
            elif power_state == 'restarted':
                if vm_power_state_current in ['paused', 'poweredon']:
                    if not module.check_mode:
                        xapi_session.xenapi.VM.hard_reboot(vm_ref)
                else:
                    module.fail_json(msg="Cannot restart VM in state '%s'!" % vm_power_state_current)
            elif power_state == 'suspended':
                if vm_power_state_current == 'poweredon':
                    if not module.check_mode:
                        xapi_session.xenapi.VM.suspend(vm_ref)
                else:
                    module.fail_json(msg="Cannot suspend VM in state '%s'!" % vm_power_state_current)
            elif power_state == 'shutdownguest':
                if vm_power_state_current == 'poweredon':
                    if not module.check_mode:
                        if timeout == 0:
                            xapi_session.xenapi.VM.clean_shutdown(vm_ref)
                        else:
                            task_ref = xapi_session.xenapi.Async.VM.clean_shutdown(vm_ref)
                            task_result = wait_for_task(module, task_ref, timeout)
                            if task_result:
                                module.fail_json(msg="Guest shutdown task failed: '%s'!" % task_result)
                else:
                    module.fail_json(msg="Cannot shutdown guest when VM is in state '%s'!" % vm_power_state_current)
            elif power_state == 'rebootguest':
                if vm_power_state_current == 'poweredon':
                    if not module.check_mode:
                        if timeout == 0:
                            xapi_session.xenapi.VM.clean_reboot(vm_ref)
                        else:
                            task_ref = xapi_session.xenapi.Async.VM.clean_reboot(vm_ref)
                            task_result = wait_for_task(module, task_ref, timeout)
                            if task_result:
                                module.fail_json(msg="Guest reboot task failed: '%s'!" % task_result)
                else:
                    module.fail_json(msg="Cannot reboot guest when VM is in state '%s'!" % vm_power_state_current)
            else:
                module.fail_json(msg="Requested VM power state '%s' is unsupported!" % power_state)
            state_changed = True
    except XenAPI.Failure as f:
        module.fail_json(msg='XAPI ERROR: %s' % f.details)
    return (state_changed, vm_power_state_resulting)