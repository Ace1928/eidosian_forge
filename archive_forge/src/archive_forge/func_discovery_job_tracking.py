from __future__ import (absolute_import, division, print_function)
import json
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import strip_substr_dict
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
def discovery_job_tracking(rest_obj, job_id, job_wait_sec):
    sleep_interval = 30
    max_retries = job_wait_sec // sleep_interval
    failed_job_status = [2070, 2100, 2101, 2102, 2103]
    success_job_status = [2060, 2020, 2090]
    job_url = (DISCOVERY_JOBS_URI + '({job_id})').format(job_id=job_id)
    loop_ctr = 0
    time.sleep(SETTLING_TIME)
    while loop_ctr < max_retries:
        loop_ctr += 1
        try:
            job_resp = rest_obj.invoke_request('GET', job_url)
            job_dict = job_resp.json_data
            job_status = job_dict['JobStatusId']
            if job_status in success_job_status:
                return JOB_TRACK_SUCCESS.format(JOB_STATUS_MAP[job_status])
            elif job_status in failed_job_status:
                return JOB_TRACK_FAIL.format(JOB_STATUS_MAP[job_status])
            time.sleep(sleep_interval)
        except HTTPError:
            return JOB_TRACK_UNABLE.format(job_id)
        except Exception as err:
            return str(err)
    return JOB_TRACK_INCOMPLETE.format(job_id, max_retries)