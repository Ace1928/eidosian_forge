from __future__ import (absolute_import, division, print_function)
import json
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import strip_substr_dict
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
def exit_discovery(module, rest_obj, job_id):
    msg = DISCOVERY_SCHEDULED
    time.sleep(SETTLING_TIME)
    djob = get_discovery_job(rest_obj, job_id)
    detailed_job = []
    if module.params.get('job_wait') and module.params.get('schedule') == 'RunNow':
        job_message = discovery_job_tracking(rest_obj, job_id, job_wait_sec=module.params['job_wait_timeout'])
        msg = job_message
        ip_details, detailed_job = get_execution_details(rest_obj, job_id)
        djob = get_discovery_job(rest_obj, job_id)
        djob.update(ip_details)
        if djob['JobStatusId'] == 2090 and (not module.params.get('ignore_partial_failure')):
            module.fail_json(msg=DISCOVERY_PARTIAL, discovery_status=djob, job_detailed_status=detailed_job)
        if djob['JobStatusId'] == 2090 and module.params.get('ignore_partial_failure'):
            module.exit_json(msg=JOB_TRACK_SUCCESS.format(JOB_STATUS_MAP[djob['JobStatusId']]), discovery_status=djob, job_detailed_status=detailed_job, changed=True)
        if ip_details.get('Failed'):
            module.fail_json(msg=JOB_TRACK_FAIL.format(JOB_STATUS_MAP[djob['JobStatusId']]), discovery_status=djob, job_detailed_status=detailed_job)
    module.exit_json(msg=msg, discovery_status=djob, job_detailed_status=detailed_job, changed=True)