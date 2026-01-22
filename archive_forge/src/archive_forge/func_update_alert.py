from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def update_alert(module, blade):
    """Update alert Watcher"""
    api_version = blade.api_version.list_versions().versions
    mod_alert = False
    try:
        alert = blade.alert_watchers.list_alert_watchers(names=[module.params['address']])
    except Exception:
        module.fail_json(msg='Failed to get information for alert email: {0}'.format(module.params['address']))
    current_state = {'enabled': alert.items[0].enabled, 'severity': alert.items[0].minimum_notification_severity}
    if current_state['enabled'] != module.params['enabled']:
        mod_alert = True
    if MIN_REQUIRED_API_VERSION in api_version:
        if current_state['severity'] != module.params['severity']:
            mod_alert = True
    if mod_alert:
        changed = True
        if not module.check_mode:
            if MIN_REQUIRED_API_VERSION in api_version:
                watcher_settings = AlertWatcher(enabled=module.params['enabled'], minimum_notification_severity=module.params['severity'])
            else:
                watcher_settings = AlertWatcher(enabled=module.params['enabled'])
            try:
                blade.alert_watchers.update_alert_watchers(names=[module.params['address']], watcher_settings=watcher_settings)
            except Exception:
                module.fail_json(msg='Failed to update alert email: {0}'.format(module.params['address']))
    else:
        changed = False
    module.exit_json(changed=changed)