from __future__ import (absolute_import, division, print_function)
import json
import time
from ansible.module_utils.common.dict_transformations import recursive_diff
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.dellemc_idrac import iDRACConnection, idrac_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import idrac_redfish_job_tracking, \
def clear_pending_bios(module, redfish_obj):
    attr = get_pending_attributes(redfish_obj)
    if not attr:
        module.exit_json(status_msg=NO_CHANGES_MSG)
    job_id, job_state = check_scheduled_bios_job(redfish_obj)
    if job_id:
        if job_state in ['Running', 'Starting']:
            module.exit_json(failed=True, status_msg=BIOS_JOB_RUNNING, job_id=job_id)
        elif job_state in ['Scheduled', 'Scheduling']:
            if module.check_mode:
                module.exit_json(status_msg=CHANGES_MSG, changed=True)
            delete_scheduled_bios_job(redfish_obj, job_id)
            module.exit_json(status_msg=SUCCESS_CLEAR, changed=True)
    if module.check_mode:
        module.exit_json(status_msg=CHANGES_MSG, changed=True)
    redfish_obj.invoke_request(CLEAR_PENDING_URI, 'POST', data='{}', dump=False)
    module.exit_json(status_msg=SUCCESS_CLEAR, changed=True)