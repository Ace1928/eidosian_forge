from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def delete_alert(module, blade):
    """Delete Alert Email"""
    changed = True
    if not module.check_mode:
        try:
            blade.alert_watchers.delete_alert_watchers(names=[module.params['address']])
        except Exception:
            module.fail_json(msg='Failed to delete alert email: {0}'.format(module.params['address']))
    module.exit_json(changed=changed)