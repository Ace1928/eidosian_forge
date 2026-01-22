from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.common.dict_transformations import recursive_diff
def assign_profile(module, rest_obj):
    mparam = module.params
    payload = {}
    if mparam.get('name'):
        prof = get_profile(rest_obj, module)
        if prof:
            payload['Id'] = prof['Id']
        else:
            module.fail_json(msg=PROFILE_NOT_FOUND.format(name=mparam.get('name')))
    target = get_target_details(module, rest_obj)
    if isinstance(target, dict):
        payload['TargetId'] = target['Id']
        if prof['ProfileState'] == 4:
            if prof['TargetId'] == target['Id']:
                module.exit_json(msg='The profile is assigned to the target {0}.'.format(target['Id']))
            else:
                module.fail_json(msg='The profile is assigned to a different target. Use the migrate command or unassign the profile and then proceed with assigning the profile to the target.')
        action = 'AssignProfile'
        msg = 'Successfully applied the assign operation.'
        try:
            resp = rest_obj.invoke_request('POST', PROFILE_ACTION.format(action='GetInvalidTargetsForAssignProfile'), data={'Id': prof['Id']})
            if target['Id'] in list(resp.json_data):
                module.fail_json(msg='The target device is invalid for the given profile.')
        except HTTPError:
            resp = None
        ad_opts_list = ['Attributes', 'Options', 'Schedule']
    else:
        if mparam.get('device_id'):
            module.fail_json(msg=target)
        action = 'AssignProfileForAutoDeploy'
        msg = 'Successfully applied the assign operation for auto-deployment.'
        payload['Identifier'] = mparam.get('device_service_tag')
        if prof['ProfileState'] == 1:
            if prof['TargetName'] == payload['Identifier']:
                module.exit_json(msg='The profile is assigned to the target {0}.'.format(payload['Identifier']))
            else:
                module.fail_json(msg='The profile is assigned to a different target. Unassign the profile and then proceed with assigning the profile to the target.')
        ad_opts_list = ['Attributes']
    boot_iso_dict = get_network_iso_payload(module)
    if boot_iso_dict:
        payload['NetworkBootToIso'] = boot_iso_dict
    ad_opts = mparam.get('attributes')
    for opt in ad_opts_list:
        if ad_opts and ad_opts.get(opt):
            attributes_check(module, rest_obj, ad_opts, prof['Id'])
            payload[opt] = ad_opts.get(opt)
    if module.check_mode:
        module.exit_json(msg=CHANGES_MSG, changed=True)
    resp = rest_obj.invoke_request('POST', PROFILE_ACTION.format(action=action), data=payload)
    res_dict = {'msg': msg, 'changed': True}
    if action == 'AssignProfile':
        try:
            res_prof = get_profile(rest_obj, module)
            time.sleep(5)
            if res_prof.get('DeploymentTaskId'):
                res_dict['job_id'] = res_prof.get('DeploymentTaskId')
                res_dict['msg'] = 'Successfully triggered the job for the assign operation.'
        except HTTPError:
            res_dict['msg'] = 'Successfully applied the assign operation. Failed to fetch job details.'
    module.exit_json(**res_dict)