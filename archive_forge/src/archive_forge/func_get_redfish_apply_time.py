from __future__ import (absolute_import, division, print_function)
import json
from ansible_collections.dellemc.openmanage.plugins.module_utils.redfish import Redfish, redfish_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import wait_for_job_completion, strip_substr_dict
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def get_redfish_apply_time(module, redfish_obj, apply_time, time_settings):
    time_set = {}
    if time_settings:
        if 'Maintenance' in apply_time:
            if apply_time not in time_settings:
                module.exit_json(failed=True, status_msg=UNSUPPORTED_APPLY_TIME.format(apply_time))
            else:
                time_set['ApplyTime'] = apply_time
                m_win = module.params.get('maintenance_window')
                validate_time(module, redfish_obj, m_win.get('start_time'))
                time_set['MaintenanceWindowStartTime'] = m_win.get('start_time')
                time_set['MaintenanceWindowDurationInSeconds'] = m_win.get('duration')
        else:
            time_set['ApplyTime'] = apply_time
    return time_set