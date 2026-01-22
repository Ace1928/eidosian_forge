from __future__ import (absolute_import, division, print_function)
import time
from datetime import datetime
import re
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def wait_for_redfish_job_complete(redfish_obj, job_uri, job_wait=True, wait_timeout=120, sleep_time=10):
    max_sleep_time = wait_timeout
    sleep_interval = sleep_time
    job_msg = 'The job is not complete after {0} seconds.'.format(wait_timeout)
    job_resp = {}
    if job_wait:
        while max_sleep_time:
            if max_sleep_time > sleep_interval:
                max_sleep_time = max_sleep_time - sleep_interval
            else:
                sleep_interval = max_sleep_time
                max_sleep_time = 0
            time.sleep(sleep_interval)
            job_resp = redfish_obj.invoke_request('GET', job_uri, api_timeout=120)
            if job_resp.json_data.get('PercentComplete') == 100:
                time.sleep(10)
                return (job_resp, '')
            if job_resp.json_data.get('JobState') == 'RebootFailed':
                time.sleep(10)
                return (job_resp, job_msg)
    else:
        time.sleep(10)
        job_resp = redfish_obj.invoke_request('GET', job_uri, api_timeout=120)
        return (job_resp, '')
    return (job_resp, job_msg)