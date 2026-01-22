from __future__ import (absolute_import, division, print_function)
import json
import os
import time
from ansible.module_utils.urls import open_url, ConnectionError, SSLValidationError
from ansible.module_utils.common.parameters import env_fallback
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import config_ipv6
def get_job_type_id(self, jobtype_name):
    """This provides an ID of the job type."""
    job_type_id = None
    resp = self.invoke_request('GET', 'JobService/JobTypes')
    data = resp.json_data['value']
    for each in data:
        if each['Name'] == jobtype_name:
            job_type_id = each['Id']
            break
    return job_type_id