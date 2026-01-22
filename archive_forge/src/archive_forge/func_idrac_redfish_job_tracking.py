from __future__ import (absolute_import, division, print_function)
import time
from datetime import datetime
import re
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def idrac_redfish_job_tracking(rest_obj, job_uri, max_job_wait_sec=600, job_state_var='JobState', job_complete_states=('Completed', 'Downloaded', 'CompletedWithErrors', 'RebootCompleted'), job_fail_states=('Failed', 'RebootFailed', 'Unknown'), job_running_states=('Running', 'RebootPending', 'Scheduling', 'Scheduled', 'Downloading', 'Waiting', 'Paused', 'New', 'PendingActivation', 'ReadyForExecution'), sleep_interval_secs=10, max_unresponsive_wait=30, initial_wait=1):
    max_retries = max_job_wait_sec // sleep_interval_secs
    unresp = max_unresponsive_wait // sleep_interval_secs
    loop_ctr = 0
    job_failed = True
    job_dict = {}
    wait_time = 0
    if set(job_complete_states) & set(job_fail_states):
        return (job_failed, 'Overlapping job states found.', job_dict, wait_time)
    msg = 'Job tracking started.'
    time.sleep(initial_wait)
    while loop_ctr < max_retries:
        loop_ctr += 1
        try:
            job_resp = rest_obj.invoke_request(job_uri, 'GET')
            job_dict = job_resp.json_data
            job_status = job_dict
            job_status = job_status.get(job_state_var, 'Unknown')
            if job_status in job_running_states:
                time.sleep(sleep_interval_secs)
                wait_time = wait_time + sleep_interval_secs
            elif job_status in job_complete_states:
                job_failed = False
                msg = 'Job tracking completed.'
                loop_ctr = max_retries
            elif job_status in job_fail_states:
                job_failed = True
                msg = 'Job is in {0} state.'.format(job_status)
                loop_ctr = max_retries
            else:
                time.sleep(sleep_interval_secs)
                wait_time = wait_time + sleep_interval_secs
        except Exception as err:
            if unresp:
                time.sleep(sleep_interval_secs)
                wait_time = wait_time + sleep_interval_secs
            else:
                job_failed = True
                msg = 'Exception in job tracking ' + str(err)
                break
            unresp = unresp - 1
    return (job_failed, msg, job_dict, wait_time)