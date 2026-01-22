from __future__ import (absolute_import, division, print_function)
import json
import time
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import (strip_substr_dict, idrac_system_reset,
from ansible.module_utils.basic import AnsibleModule
def configure_boot_options(module, idrac, res_id, payload):
    is_job, progress_job = get_scheduled_job(idrac)
    job_data, job_wait = ({}, module.params['job_wait'])
    resp_data = get_response_attributes(module, idrac, res_id)
    override_mode = resp_data['BootSourceOverrideMode']
    if module.params['reset_type'] == 'none':
        job_wait = False
    if is_job:
        module.fail_json(msg=JOB_EXISTS)
    boot_seq_resp = idrac.invoke_request(BOOT_SEQ_URI.format(res_id), 'GET')
    seq_key = 'BootSeq' if override_mode == 'Legacy' else 'UefiBootSeq'
    boot_seq_data = boot_seq_resp.json_data['Attributes'][seq_key]
    [each.update({'Enabled': payload.get(each['Name'])}) for each in boot_seq_data if payload.get(each['Name']) is not None]
    seq_payload = {'Attributes': {seq_key: boot_seq_data}, '@Redfish.SettingsApplyTime': {'ApplyTime': 'OnReset'}}
    if seq_key == 'UefiBootSeq':
        for i in range(len(boot_seq_data)):
            if payload.get(resp_data['BootOrder'][i]) is not None:
                boot_seq_data[i].update({'Enabled': payload.get(resp_data['BootOrder'][i])})
        seq_payload['Attributes'][seq_key] = boot_seq_data
    resp = idrac.invoke_request(PATCH_BOOT_SEQ_URI.format(res_id), 'PATCH', data=seq_payload)
    if resp.status_code == 202:
        location = resp.headers['Location']
        job_id = location.split('/')[-1]
        reset, track_failed, reset_msg, reset_job_resp = system_reset(module, idrac, res_id)
        if reset_job_resp:
            job_data = reset_job_resp.json_data
        if reset:
            job_resp, error_msg = wait_for_idrac_job_completion(idrac, JOB_URI_ID.format(job_id), job_wait=job_wait, wait_timeout=module.params['job_wait_timeout'])
            if error_msg:
                module.fail_json(msg=error_msg)
            job_data = job_resp.json_data
        else:
            module.fail_json(msg=reset_msg)
    return job_data