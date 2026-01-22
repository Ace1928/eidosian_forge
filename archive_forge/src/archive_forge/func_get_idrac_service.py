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
def get_idrac_service(idrac, res_id):
    srvc = IDRAC_SERVICE.format(res_id=res_id)
    resp = idrac.invoke_request(f'{MANAGERS_URI}/{res_id}', 'GET')
    srvc_data = resp.json_data
    dell_srvc = srvc_data['Links']['Oem']['Dell']['DelliDRACCardService']
    srvc = dell_srvc.get('@odata.id', IDRAC_SERVICE.format(res_id=res_id))
    return srvc