from __future__ import (absolute_import, division, print_function)
import json
import base64
import os
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import reset_idrac
def get_actions_map(idrac, idrac_service_uri):
    actions = idrac_service_actions
    try:
        resp = idrac.invoke_request(idrac_service_uri, 'GET')
        srvc_data = resp.json_data
        actions = dict(((k, v.get('target')) for k, v in srvc_data.get('Actions').items()))
    except Exception:
        actions = idrac_service_actions
    return actions