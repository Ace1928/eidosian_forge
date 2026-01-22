from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.compat.version import LooseVersion
def track_compliance_task_completion(rest_obj, baseline_identifier_val, module):
    """
    wait for the compliance configuration task to complete
    """
    baseline_info = get_baseline_compliance_info(rest_obj, baseline_identifier_val)
    command = module.params['command']
    if module.params.get('job_wait'):
        wait_time = 5
        retries_count_limit = module.params['job_wait_timeout'] / wait_time
        retries_count = 0
        time.sleep(wait_time)
        if command == 'create':
            msg = CREATE_MSG
        else:
            msg = MODIFY_MSG
        while retries_count <= retries_count_limit:
            if baseline_info['PercentageComplete'] == '100':
                break
            retries_count += 1
            time.sleep(wait_time)
            baseline_info = get_baseline_compliance_info(rest_obj, baseline_identifier_val)
        if baseline_info['PercentageComplete'] != '100':
            msg = TASK_PROGRESS_MSG
    else:
        msg = TASK_PROGRESS_MSG
    return (msg, baseline_info)