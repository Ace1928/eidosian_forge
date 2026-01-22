from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.urls import ConnectionError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def spawn_update_job(rest_obj, job_payload):
    """Spawns an update job and tracks it to completion."""
    job_uri, job_details = ('JobService/Jobs', {})
    job_resp = rest_obj.invoke_request('POST', job_uri, data=job_payload)
    if job_resp.status_code == 201:
        job_details = job_resp.json_data
    return job_details