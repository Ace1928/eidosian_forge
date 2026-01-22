from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
def is_maint_flag_enabled(self):
    feature_flags = self._exec('rabbitmqctl', ['list_feature_flags', '-q'], True)
    for param_item in feature_flags:
        name, state = param_item.split('\t')
        if name == 'maintenance_mode_status' and state == 'enabled':
            return True
    return False