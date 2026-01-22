from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.service import fail_if_missing
from ansible.module_utils.common.text.converters import to_native
def handle_state(module, result, service_path):
    """Set service running state as needed.

    Takes into account the fact that a service may not be loaded (no supervise directory) in
    which case it is 'stopped' as far as the service manager is concerned. No status information
    can be obtained and the service can only be 'started'.
    """
    result['state'] = module.params['state']
    state = module.params['state']
    action = None
    enabled = result['enabled']
    if not service_is_loaded(module, service_path):
        if state in ['started', 'restarted', 'reloaded']:
            action = 'start'
            result['state'] = 'started'
        elif state == 'reset':
            if enabled:
                action = 'start'
                result['state'] = 'started'
            else:
                result['state'] = None
        else:
            result['state'] = None
    else:
        result['status'] = get_service_status(module, service_path)
        running = service_is_running(result['status'])
        if state == 'started':
            if not running:
                action = 'start'
        elif state == 'stopped':
            if running:
                action = 'stop'
        elif state == 'reset':
            if enabled is not running:
                if running:
                    action = 'stop'
                    result['state'] = 'stopped'
                else:
                    action = 'start'
                    result['state'] = 'started'
        elif state == 'restarted':
            if not running:
                action = 'start'
                result['state'] = 'started'
            else:
                action = 'condrestart'
        elif state == 'reloaded':
            if not running:
                action = 'start'
                result['state'] = 'started'
            else:
                action = 'hangup'
    if action:
        result['changed'] = True
        if not module.check_mode:
            rc, out, err = run_sys_ctl(module, [action, service_path])
            if rc != 0:
                module.fail_json(msg='Unable to %s service %s: %s' % (action, service_path, err))