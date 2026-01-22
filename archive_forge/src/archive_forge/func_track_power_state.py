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
def track_power_state(redfish_obj, desired_state, retries=POWER_CHECK_RETRIES, interval=POWER_CHECK_INTERVAL):
    count = retries
    while count:
        ps = get_power_state(redfish_obj)
        if ps in desired_state:
            achieved = True
            break
        else:
            time.sleep(interval)
        count = count - 1
    else:
        achieved = False
    return achieved