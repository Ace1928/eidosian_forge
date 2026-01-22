from __future__ import absolute_import, division, print_function
import json
import time
from urllib.error import HTTPError, URLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import (
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import (
def perform_operation_for_main(idrac, module, obj, diff, _invalid_attr):
    job_wait_timeout = module.params.get('job_wait_timeout')
    if diff:
        if module.check_mode:
            module.exit_json(msg=CHANGES_FOUND_MSG, changed=True, invalid_attributes=_invalid_attr)
        else:
            job_resp, invalid_attr, job_wait = obj.perform_operation()
            job_dict = {}
            if (job_tracking_uri := job_resp.headers.get('Location')):
                job_id = job_tracking_uri.split('/')[-1]
                job_uri = iDRAC_JOB_URI.format(job_id=job_id)
                if job_wait:
                    job_failed, msg, job_dict, wait_time = idrac_redfish_job_tracking(idrac, job_uri, max_job_wait_sec=job_wait_timeout, sleep_interval_secs=1)
                    job_dict = remove_key(job_dict, regex_pattern='(.*?)@odata')
                    if int(wait_time) >= int(job_wait_timeout):
                        module.exit_json(msg=WAIT_TIMEOUT_MSG.format(job_wait_timeout), changed=True, job_status=job_dict)
                    if job_failed:
                        module.fail_json(msg=job_dict.get('Message'), invalid_attributes=invalid_attr, job_status=job_dict)
                else:
                    job_resp = idrac.invoke_request(job_uri, 'GET')
                    job_dict = job_resp.json_data
                    job_dict = remove_key(job_dict, regex_pattern='(.*?)@odata')
            if job_dict.get('JobState') == 'Completed':
                firm_ver = get_idrac_firmware_version(idrac)
                msg = SUCCESS_MSG if not invalid_attr else VALID_AND_INVALID_ATTR_MSG
                if LooseVersion(firm_ver) < '3.0' and isinstance(obj, OEMNetworkAttributes):
                    message_id = job_dict.get('MessageId')
                    if message_id == 'SYS053':
                        module.exit_json(msg=msg, changed=True, job_status=job_dict)
                    elif message_id == 'SYS055':
                        module.exit_json(msg=VALID_AND_INVALID_ATTR_MSG, changed=True, job_status=job_dict)
                    elif message_id == 'SYS067':
                        module.fail_json(msg=INVALID_ATTR_MSG, job_status=job_dict)
                    else:
                        module.fail_json(msg=job_dict.get('Message'))
            else:
                msg = SCHEDULE_MSG
            module.exit_json(msg=msg, invalid_attributes=invalid_attr, job_status=job_dict, changed=True)
    else:
        if module.check_mode:
            module.exit_json(msg=NO_CHANGES_FOUND_MSG, invalid_attributes=_invalid_attr)
        elif _invalid_attr:
            job_resp, invalid_attr, job_wait = obj.perform_operation()
        module.exit_json(msg=NO_CHANGES_FOUND_MSG, invalid_attributes=_invalid_attr)