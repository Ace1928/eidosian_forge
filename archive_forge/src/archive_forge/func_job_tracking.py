from __future__ import (absolute_import, division, print_function)
import json
import os
import time
from ansible.module_utils.urls import open_url, ConnectionError, SSLValidationError
from ansible.module_utils.common.parameters import env_fallback
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import config_ipv6
def job_tracking(self, job_id, job_wait_sec=600, sleep_time=60):
    """
        job_id: job id
        job_wait_sec: Maximum time to wait to fetch the final job details in seconds
        sleep_time: Maximum time to sleep in seconds in each job details fetch
        """
    max_sleep_time = job_wait_sec
    sleep_interval = sleep_time
    while max_sleep_time:
        if max_sleep_time > sleep_interval:
            max_sleep_time = max_sleep_time - sleep_interval
        else:
            sleep_interval = max_sleep_time
            max_sleep_time = 0
        time.sleep(sleep_interval)
        exit_poll, job_failed, job_message = self.get_job_info(job_id)
        if exit_poll is True:
            return (job_failed, job_message)
    return (True, 'The job is not complete after {0} seconds.'.format(job_wait_sec))