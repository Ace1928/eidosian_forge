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
def reset_bios(module, redfish_obj):
    attr = get_pending_attributes(redfish_obj)
    if attr:
        module.exit_json(status_msg=BIOS_RESET_PENDING, failed=True)
    if module.check_mode:
        module.exit_json(status_msg=CHANGES_MSG, changed=True)
    redfish_obj.invoke_request(RESET_BIOS_DEFAULT, 'POST', data='{}', dump=True)
    reset_success = reset_host(module, redfish_obj)
    if not reset_success:
        module.exit_json(failed=True, status_msg='{0} {1}'.format(RESET_TRIGGERRED, HOST_RESTART_FAILED))
    log_msg = track_log_entry(redfish_obj)
    module.exit_json(status_msg=log_msg, changed=True)