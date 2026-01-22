from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import strip_substr_dict, job_tracking
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import CHANGES_MSG, NO_CHANGES_MSG
def job_wait(module, rest_obj, job):
    mparams = module.params
    if mparams.get('job_schedule') != 'startnow':
        module.exit_json(changed=True, msg=JOB_SCHEDULED, job=strip_substr_dict(job))
    if not module.params.get('job_wait'):
        module.exit_json(changed=True, msg=APPLY_TRIGGERED, job=strip_substr_dict(job))
    else:
        job_msg = SUCCESS_MSG
        job_failed, msg, job_dict, wait_time = job_tracking(rest_obj, JOB_URI.format(job_id=job['Id']), max_job_wait_sec=module.params.get('job_wait_timeout'), initial_wait=3)
        if job_failed:
            try:
                job_resp = rest_obj.invoke_request('GET', LAST_EXEC.format(job_id=job['Id']))
                msg = job_resp.json_data.get('Value')
                job_msg = msg.replace('\n', ' ')
            except Exception:
                job_msg = msg
        module.exit_json(failed=job_failed, msg=job_msg, job=strip_substr_dict(job), changed=True)