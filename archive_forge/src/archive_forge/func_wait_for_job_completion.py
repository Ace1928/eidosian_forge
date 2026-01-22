from __future__ import (absolute_import, division, print_function)
import json
import re
import time
import os
from ansible.module_utils.urls import open_url, ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.common.parameters import env_fallback
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import config_ipv6
def wait_for_job_completion(self, job_uri, job_wait=False, reboot=False, apply_update=False):
    """
        This function wait till the job completion.
        :param job_uri: uri to track job.
        :param job_wait: True or False decide whether to wait till the job completion.
        :return: object
        """
    time.sleep(5)
    response = self.invoke_request(job_uri, 'GET')
    while job_wait:
        response = self.invoke_request(job_uri, 'GET')
        if response.json_data.get('PercentComplete') == 100 and response.json_data.get('JobState') == 'Completed':
            break
        if response.json_data.get('JobState') == 'Starting' and (not reboot) and apply_update:
            break
        time.sleep(30)
    return response