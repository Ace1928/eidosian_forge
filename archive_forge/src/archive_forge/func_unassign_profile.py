from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.common.dict_transformations import recursive_diff
def unassign_profile(module, rest_obj):
    mparam = module.params
    prof = {}
    if mparam.get('name'):
        payload = {}
        prof = get_profile(rest_obj, module)
        if prof:
            if prof['ProfileState'] == 0:
                module.exit_json(msg='Profile is in an unassigned state.')
            if prof['DeploymentTaskId']:
                try:
                    resp = rest_obj.invoke_request('GET', JOB_URI.format(job_id=prof['DeploymentTaskId']))
                    job_dict = resp.json_data
                    job_status = job_dict.get('LastRunStatus')
                    if job_status.get('Name') == 'Running':
                        module.fail_json(msg='Profile deployment task is in progress. Wait for the job to finish.')
                except HTTPError:
                    msg = 'Unable to fetch job details. Applied the unassign operation'
            payload['ProfileIds'] = [prof['Id']]
        else:
            module.fail_json(msg=PROFILE_NOT_FOUND.format(name=mparam.get('name')))
    if mparam.get('filters'):
        payload = mparam.get('filters')
    if module.check_mode:
        module.exit_json(msg=CHANGES_MSG, changed=True)
    msg = 'Successfully applied the unassign operation. No job was triggered.'
    resp = rest_obj.invoke_request('POST', PROFILE_ACTION.format(action='UnassignProfiles'), data=payload)
    res_dict = {'msg': msg, 'changed': True}
    try:
        res_prof = get_profile(rest_obj, module)
        time.sleep(3)
        if res_prof.get('DeploymentTaskId'):
            res_dict['job_id'] = res_prof.get('DeploymentTaskId')
            res_dict['msg'] = 'Successfully triggered a job for the unassign operation.'
    except HTTPError:
        res_dict['msg'] = 'Successfully triggered a job for the unassign operation. Failed to fetch the job details.'
    module.exit_json(**res_dict)