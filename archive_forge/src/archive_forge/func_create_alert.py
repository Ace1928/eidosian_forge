from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def create_alert(module, blade):
    """Create Alert Email"""
    changed = True
    if not module.check_mode:
        api_version = blade.api_version.list_versions().versions
        if MIN_REQUIRED_API_VERSION in api_version:
            watcher_settings = AlertWatcher(minimum_notification_severity=module.params['severity'])
            try:
                blade.alert_watchers.create_alert_watchers(names=[module.params['address']], watcher_settings=watcher_settings)
            except Exception:
                module.fail_json(msg='Failed to create alert email: {0}'.format(module.params['address']))
        else:
            try:
                blade.alert_watchers.create_alert_watchers(names=[module.params['address']])
            except Exception:
                module.fail_json(msg='Failed to create alert email: {0}'.format(module.params['address']))
        if not module.params['enabled']:
            watcher_settings = AlertWatcher(enabled=module.params['enabled'])
            try:
                blade.alert_watchers.update_alert_watchers(names=[module.params['address']], watcher_settings=watcher_settings)
            except Exception:
                module.fail_json(msg='Failed to disable during create alert email: {0}'.format(module.params['address']))
    module.exit_json(changed=changed)