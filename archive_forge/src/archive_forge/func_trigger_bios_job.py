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
def trigger_bios_job(redfish_obj):
    job_id = None
    payload = {'TargetSettingsURI': BIOS_SETTINGS}
    resp = redfish_obj.invoke_request(IDRAC_JOBS_URI, 'POST', data=payload)
    job_id = resp.headers['Location'].split('/')[-1]
    return job_id